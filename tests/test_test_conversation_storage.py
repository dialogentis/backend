import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy.exc import IntegrityError

from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.runtime.checkpoint import (
    CheckpointIdentity,
    decrypt_checkpoint,
    encrypt_checkpoint,
)
from dialogentis.runtime.core import RuntimeState
from dialogentis.runtime.models import TestConversation as ConversationRecord
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_encrypted_state_rls_constraints_and_tenant_deletion(app):
    async with client_for(app) as owner, client_for(app) as stranger:
        user_id, org_id, agent_id = await create_agent(app, owner)
        other_user, other_org, other_agent = await create_agent(app, stranger)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "checkpoint-key")).version
        version_id, conversation_id = version.id, uuid4()
        identity = CheckpointIdentity(org_id, agent_id, version_id, conversation_id)
        settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
        state = RuntimeState(
            version_id=version_id,
            flow_hash="a" * 64,
            node_id=uuid4(),
            status="waiting_input",
            turn=0,
            fallback_count=0,
            slot_values=(("secret", "DO_NOT_STORE_PLAIN"),),
        )
        ciphertext = encrypt_checkpoint(state, identity, settings)
        now = datetime.now(UTC)

        def record(**changes):
            return ConversationRecord(
                **(
                    {
                        "id": uuid4(),
                        "organization_id": org_id,
                        "agent_id": agent_id,
                        "version_id": version_id,
                        "created_by": user_id,
                        "checkpoint": ciphertext,
                        "revision": 1,
                        "idle_expires_at": now + timedelta(minutes=30),
                        "absolute_expires_at": now + timedelta(hours=23),
                    }
                    | changes
                )
            )

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(record(id=conversation_id))
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            saved = await db.get(ConversationRecord, conversation_id)
            assert saved is not None and saved.checkpoint is not None
            assert b"DO_NOT_STORE_PLAIN" not in saved.checkpoint
            assert decrypt_checkpoint(saved.checkpoint, identity, settings) == state
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(ConversationRecord, conversation_id) is None
        async with unit_of_work(app.state.session_factory) as db:
            assert await db.get(ConversationRecord, conversation_id) is None
        for bad in (
            {"agent_id": other_agent},
            {"checkpoint": b"plain text"},
            {"absolute_expires_at": now + timedelta(days=2)},
            {"revision": 0},
        ):
            with pytest.raises(IntegrityError):
                async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                    db.add(record(**bad))
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(ConversationRecord, conversation_id) is None
        assert other_user != user_id
