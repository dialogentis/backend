import json
import subprocess
import sys
from dataclasses import replace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from dialogentis.config import Settings
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.model_store import (
    InvalidModelArtifact,
    ModelIdentity,
    pack_model,
    unpack_model,
)
from tests.test_nlu_intents import training_dataset
from tests.test_nlu_learned_entities import learned_dataset


@pytest.fixture
def model_data():
    data, _ = learned_dataset()
    identity = ModelIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    settings = Settings(encryption_key=Fernet.generate_key().decode(), encryption_key_version=3)
    adapter = NLUAdapter.train(identity.version_id, data, Thresholds())
    return adapter, identity, settings


def test_signed_roundtrip_retains_predictions_and_report(model_data):
    adapter, identity, settings = model_data
    content = pack_model(adapter, identity, settings)
    restored = unpack_model(content, identity, settings)
    text = "👩🏽‍💻 доставьте в Москва номер 8"
    assert restored.predict(text) == adapter.predict(text)
    data, _ = learned_dataset()
    data.examples[0].text += " изменено"
    with pytest.raises(ValueError, match="training data"):
        restored.evaluate(data)


def test_roundtrip_without_learned_entities():
    data = training_dataset()
    identity = ModelIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    settings = Settings(encryption_key=Fernet.generate_key().decode())
    adapter = NLUAdapter.train(identity.version_id, data, Thresholds())
    assert adapter.entity_model is None
    restored = unpack_model(pack_model(adapter, identity, settings), identity, settings)
    assert restored.predict("где заказ 0") == adapter.predict("где заказ 0")


def test_tampered_wrong_identity_and_incompatible_models_never_unpickle(model_data, monkeypatch):
    adapter, identity, settings = model_data
    content = pack_model(adapter, identity, settings)
    called = []
    monkeypatch.setattr("dialogentis.nlu.model_store.pickle.loads", lambda _: called.append(True))
    tampered = bytearray(content)
    tampered[-33] ^= 1
    for invalid, target in [
        (bytes(tampered), identity),
        (content, replace(identity, organization_id=uuid4())),
        (content, replace(identity, agent_id=uuid4())),
        (content, replace(identity, version_id=uuid4())),
        (content, replace(identity, artifact_id=uuid4())),
        (content[:20], identity),
    ]:
        with pytest.raises(InvalidModelArtifact):
            unpack_model(invalid, target, settings)
    with pytest.raises(InvalidModelArtifact):
        unpack_model(content, identity, settings.model_copy(update={"encryption_key_version": 4}))
    with pytest.raises(InvalidModelArtifact):
        unpack_model(content, identity, Settings(encryption_key=Fernet.generate_key().decode()))
    monkeypatch.setattr(
        "dialogentis.nlu.model_store._environment", lambda: {"python": "incompatible"}
    )
    with pytest.raises(InvalidModelArtifact):
        unpack_model(content, identity, settings)
    assert not called


def test_model_size_and_missing_key_are_rejected(model_data):
    adapter, identity, settings = model_data
    with pytest.raises(InvalidModelArtifact, match="size limit"):
        pack_model(adapter, identity, settings.model_copy(update={"max_model_artifact_bytes": 100}))
    with pytest.raises(InvalidModelArtifact, match="key unavailable"):
        pack_model(adapter, identity, settings.model_copy(update={"encryption_key": None}))
    with pytest.raises(InvalidModelArtifact, match="identity mismatch"):
        pack_model(adapter, replace(identity, version_id=uuid4()), settings)


def test_model_loads_in_fresh_python_process(model_data, tmp_path):
    adapter, identity, settings = model_data
    artifact_path = tmp_path / "model"
    artifact_path.write_bytes(pack_model(adapter, identity, settings))
    script = """
import json, sys
from pathlib import Path
from uuid import UUID
from dialogentis.config import Settings
from dialogentis.nlu.model_store import ModelIdentity, unpack_model
data = json.loads(sys.stdin.read())
identity = ModelIdentity(*(UUID(value) for value in data['identity']))
settings = Settings(encryption_key=data['key'], encryption_key_version=3)
model = unpack_model(Path(data['path']).read_bytes(), identity, settings)
result = model.predict(data['text'])
print(json.dumps({
    'version': str(result.version_id),
    'intent': str(result.intent.intent_id),
    'scores': [item.score for item in result.intent.ranking],
    'entities': [(str(item.entity_id), item.start, item.end, item.value)
                 for item in result.entities.matches],
}, ensure_ascii=False))
"""
    text = "👩🏽‍💻 доставьте в Москва номер 8"
    assert settings.encryption_key is not None
    data = {
        "identity": [
            str(identity.organization_id),
            str(identity.agent_id),
            str(identity.version_id),
            str(identity.artifact_id),
        ],
        "key": settings.encryption_key.get_secret_value(),
        "path": str(artifact_path),
        "text": text,
    }
    process = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(process.stdout)
    expected = adapter.predict(text)
    assert result["version"] == str(identity.version_id)
    assert result["intent"] == str(expected.intent.intent_id)
    assert result["scores"] == pytest.approx([item.score for item in expected.intent.ranking])
    assert result["entities"] == [
        [str(item.entity_id), item.start, item.end, item.value]
        for item in expected.entities.matches
    ]
