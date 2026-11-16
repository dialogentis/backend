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


async def test_scheduled_expiry_cleanup_respects_locks_and_dispatcher_permissions(app):
    from pathlib import Path

    from sqlalchemy import delete, select
    from sqlalchemy.exc import DBAPIError

    from dialogentis.jobs.dispatcher import dispatcher_factory
    from dialogentis.jobs.maintenance import reconcile_once
    from dialogentis.jobs.models import OutboxEvent
    from dialogentis.runtime.cleanup import cleanup_expired_conversations

    async with client_for(app) as client:
        user_id, org_id, agent_id = await create_agent(app, client)
        await populate(app, org_id, agent_id)
        version_id = (await prepare(app, user_id, org_id, agent_id, 1, "expiry-key")).version.id
        now = datetime.now(UTC)
        expired_id, live_id = uuid4(), uuid4()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            for item_id, expiry in (
                (expired_id, now - timedelta(seconds=1)),
                (live_id, now + timedelta(minutes=30)),
            ):
                db.add(
                    ConversationRecord(
                        id=item_id,
                        organization_id=org_id,
                        agent_id=agent_id,
                        version_id=version_id,
                        created_by=user_id,
                        checkpoint=None,
                        revision=1,
                        idle_expires_at=expiry,
                        absolute_expires_at=now + timedelta(hours=23),
                    )
                )
            await db.execute(delete(OutboxEvent).where(OutboxEvent.organization_id == org_id))
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            async with dispatcher() as db:
                assert (
                    org_id
                    in (
                        await db.scalars(select(ConversationRecord.organization_id).distinct())
                    ).all()
                )
            with pytest.raises(DBAPIError):
                async with dispatcher() as db:
                    await db.execute(select(ConversationRecord.checkpoint))
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as lock_db:
                await lock_db.scalar(
                    select(ConversationRecord)
                    .where(ConversationRecord.id == expired_id)
                    .with_for_update()
                )
                async with unit_of_work(
                    app.state.session_factory, organization_id=org_id
                ) as cleanup_db:
                    assert await cleanup_expired_conversations(cleanup_db, org_id) == 0
            window = datetime(2041, 1, 1, tzinfo=UTC)
            assert await reconcile_once(app.state.session_factory, dispatcher, window, frozenset())
            assert not await reconcile_once(
                app.state.session_factory, dispatcher, window, frozenset()
            )
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                assert await db.get(ConversationRecord, expired_id) is None
                assert await db.get(ConversationRecord, live_id) is not None
        finally:
            await dispatcher.kw["bind"].dispose()
