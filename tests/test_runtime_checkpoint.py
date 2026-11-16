from dataclasses import replace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from dialogentis.config import Settings
from dialogentis.runtime.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    decrypt_checkpoint,
    encrypt_checkpoint,
)
from dialogentis.runtime.core import RuntimeState


def test_encrypted_checkpoint_binds_all_context_and_hides_values():
    settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
    identity = CheckpointIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    state = RuntimeState(
        version_id=identity.version_id,
        flow_hash="a" * 64,
        node_id=uuid4(),
        status="waiting_input",
        turn=2,
        fallback_count=0,
        slot_values=(("secret", "PRIVATE_SENTINEL"),),
    )
    encrypted = encrypt_checkpoint(state, identity, settings)
    assert b"PRIVATE_SENTINEL" not in encrypted and b"secret" not in encrypted
    assert encrypted != encrypt_checkpoint(state, identity, settings)
    assert decrypt_checkpoint(encrypted, identity, settings) == state
    for field in ("organization_id", "agent_id", "version_id", "conversation_id"):
        with pytest.raises(CheckpointError, match="^invalid_checkpoint$"):
            decrypt_checkpoint(encrypted, replace(identity, **{field: uuid4()}), settings)
    with pytest.raises(CheckpointError):
        encrypt_checkpoint(state, replace(identity, version_id=uuid4()), settings)
    tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
    with pytest.raises(CheckpointError, match="^invalid_checkpoint$"):
        decrypt_checkpoint(tampered, identity, settings)
    with pytest.raises(CheckpointError):
        decrypt_checkpoint(
            encrypted, identity, Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
        )
    with pytest.raises(CheckpointError):
        decrypt_checkpoint(
            encrypted, identity, settings.model_copy(update={"encryption_key_version": 2})
        )
    with pytest.raises(CheckpointError):
        decrypt_checkpoint(b"broken", identity, settings)
    with pytest.raises(CheckpointError):
        encrypt_checkpoint(state, identity, settings.model_copy(update={"encryption_key": None}))


def test_roundtrip_resumes_real_runtime_with_sensitive_values():
    from dialogentis.runtime.core import Prediction, RuntimeEngine
    from tests.test_runtime_ask_slot import ask_config

    identity = CheckpointIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
    config, _, _ = ask_config()
    engine = RuntimeEngine(
        identity.version_id, config, lambda _: Prediction(identity.version_id, None, ("low_score",))
    )
    pending = engine.start()
    collected = engine.send(pending.state, "731598")
    sealed = encrypt_checkpoint(collected.state, identity, settings)
    assert b"731598" not in sealed
    resumed = engine.send(decrypt_checkpoint(sealed, identity, settings), "следующий запрос")
    assert resumed.state.status == "waiting_input"
    assert dict(resumed.state.slot_values)["order_id"] == 731598
    assert "731598" not in repr(resumed.trace) and "731598" not in repr(resumed.state)
