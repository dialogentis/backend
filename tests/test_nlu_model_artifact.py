import hashlib
import os
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import ArtifactUnavailable, reserve_artifact
from dialogentis.artifacts import ArtifactIntegrityError
from dialogentis.db import unit_of_work
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.model_store import InvalidModelArtifact, ModelIdentity, load_model, store_model
from tests.test_artifact_service import context as context
from tests.test_nlu_learned_entities import learned_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_model_storage_is_tenant_scoped_and_checks_artifact_type_and_checksum(context):
    factory, storage, settings, first, second = context
    settings = settings.model_copy(
        update={
            "artifact_quota_bytes": 2 * 1024 * 1024,
            "encryption_key": SecretStr(Fernet.generate_key().decode()),
        }
    )
    data, _ = learned_dataset()
    agent_id, version_id = uuid4(), uuid4()
    adapter = NLUAdapter.train(version_id, data, Thresholds())
    async with unit_of_work(factory, organization_id=first) as db:
        artifact = await reserve_artifact(
            db, first, kind="model", max_bytes=1024 * 1024, settings=settings
        )
        temporary = await reserve_artifact(
            db, first, kind="temporary", max_bytes=10, settings=settings
        )
        artifact_id, temporary_id = artifact.id, temporary.id
    identity = ModelIdentity(first, agent_id, version_id, artifact_id)
    async with unit_of_work(factory, organization_id=first) as db:
        with pytest.raises(ArtifactUnavailable):
            await store_model(
                db,
                storage,
                ModelIdentity(first, agent_id, version_id, temporary_id),
                adapter,
                settings,
            )
        stored = await store_model(db, storage, identity, adapter, settings)
        assert stored.status == "ready" and stored.kind == "model" and stored.expires_at is None
    async with unit_of_work(factory, organization_id=first) as db:
        restored = await load_model(db, storage, identity, settings)
        text = "👩🏽‍💻 доставьте в Москва номер 8"
        assert restored.predict(text) == adapter.predict(text)
        # Lowering the aggregate quota must not disable existing models.
        reduced = settings.model_copy(update={"artifact_quota_bytes": 1})
        assert (await load_model(db, storage, identity, reduced)).predict(text) == adapter.predict(
            text
        )
    async with unit_of_work(factory, organization_id=second) as db:
        with pytest.raises(ArtifactUnavailable):
            await load_model(db, storage, identity, settings)
        with pytest.raises(ArtifactUnavailable):
            await load_model(
                db, storage, ModelIdentity(second, agent_id, version_id, artifact_id), settings
            )
    path = storage.root / str(first) / str(artifact_id)
    content = bytearray(path.read_bytes())
    content[-33] ^= 1
    path.write_bytes(content)
    async with unit_of_work(factory, organization_id=first) as db:
        with pytest.raises(ArtifactIntegrityError):
            await load_model(db, storage, identity, settings)
    async with unit_of_work(factory, organization_id=first) as db:
        artifact = await db.get(Artifact, artifact_id)
        assert artifact is not None
        artifact.sha256 = hashlib.sha256(content).hexdigest()
    async with unit_of_work(factory, organization_id=first) as db:
        with pytest.raises(InvalidModelArtifact, match="authentication failed"):
            await load_model(db, storage, identity, settings)
