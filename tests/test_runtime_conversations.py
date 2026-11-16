import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import select

from dialogentis.artifact_service import reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.jobs.models import Job
from dialogentis.models import Membership, User
from dialogentis.runtime.conversations import (
    commit_turn,
    create_conversation,
    read_conversation,
    reserve_turn,
)
from dialogentis.runtime.core import RuntimeState
from dialogentis.runtime.models import TestConversation as ConversationRecord
from dialogentis.versions.models import AgentVersion
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_conversation_lifecycle_concurrency_expiry_and_permissions(app):
    async with client_for(app) as owner, client_for(app) as other:
        user_id, org_id, agent_id = await create_agent(app, owner)
        other_id, _, _ = await create_agent(app, other)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "conversation-service")).version
        version_id = version.id
        settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
        initial = RuntimeState(
            version_id=version_id,
            flow_hash="a" * 64,
            node_id=uuid4(),
            status="waiting_input",
            turn=0,
            fallback_count=0,
            slot_values=(("secret", "SENSITIVE_SENTINEL"),),
        )

        async def user(db, uid):
            result = await db.get(User, uid)
            assert result is not None
            return result

        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            with pytest.raises(HTTPException) as unready:
                await create_conversation(
                    db, await user(db, user_id), org_id, agent_id, version_id, initial, settings
                )
            assert unready.value.status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            artifact = await reserve_artifact(
                db, org_id, kind="model", max_bytes=5, settings=app.state.settings
            )
            await write_artifact(
                db,
                ArtifactStorage(app.state.settings.artifact_root),
                org_id,
                artifact.id,
                [b"model"],
            )
            saved_version = await db.get(AgentVersion, version_id)
            assert saved_version is not None
            job = await db.get(Job, saved_version.job_id)
            assert job is not None
            saved_version.model_artifact_id, saved_version.validation_report = artifact.id, {}
            job.status = "succeeded"
            db.add(Membership(organization_id=org_id, user_id=other_id, role="viewer"))
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            created = await create_conversation(
                db, await user(db, user_id), org_id, agent_id, version_id, initial, settings
            )
            credential = created.credential
            assert credential is not None
            conversation_id = created.id
            assert created.revision == 1 and "SENSITIVE_SENTINEL" not in repr(created)
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=other_id
        ) as db:
            with pytest.raises(HTTPException) as forbidden:
                await read_conversation(
                    db,
                    await user(db, other_id),
                    org_id,
                    conversation_id,
                    settings,
                    credential,
                )
            assert forbidden.value.status_code == 404
            own = await create_conversation(
                db, await user(db, other_id), org_id, agent_id, version_id, initial, settings
            )
            own_credential = own.credential
            assert own_credential is not None
            assert own.id != conversation_id
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            saved = await db.get(ConversationRecord, conversation_id)
            assert (
                saved is not None
                and saved.checkpoint is not None
                and b"SENSITIVE_SENTINEL" not in saved.checkpoint
            )
            assert saved.credential_hash is not None and saved.credential_hash != credential
            assert credential not in repr(created)
            for invalid in ("", "x" * 43, own_credential, "☃" * 43, credential + "x"):
                with pytest.raises(HTTPException) as invalid_token:
                    await read_conversation(
                        db, await user(db, user_id), org_id, conversation_id, settings, invalid
                    )
                assert invalid_token.value.status_code == 404
            assert (
                await read_conversation(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    settings,
                    credential,
                )
            ).state == initial
        next_state = initial.model_copy(update={"turn": 1})

        async def reserve():
            async with unit_of_work(
                app.state.session_factory, organization_id=org_id, user_id=user_id
            ) as db:
                return await reserve_turn(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    1,
                    settings,
                    credential,
                )

        outcomes = await asyncio.gather(reserve(), reserve(), return_exceptions=True)
        assert (
            sum(isinstance(item, HTTPException) and item.status_code == 409 for item in outcomes)
            == 1
        )
        lease = next(item for item in outcomes if not isinstance(item, BaseException))
        assert lease.revision == 1 and "SENSITIVE_SENTINEL" not in repr(lease)
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            with pytest.raises(HTTPException) as stale:
                await commit_turn(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    1,
                    uuid4(),
                    next_state,
                    settings,
                    credential,
                )
            assert stale.value.status_code == 409
            result = await commit_turn(
                db,
                await user(db, user_id),
                org_id,
                conversation_id,
                1,
                lease.token,
                next_state,
                settings,
                credential,
            )
            assert result.revision == 2
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            row = await db.get(ConversationRecord, conversation_id)
            assert row is not None
            row.idle_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            with pytest.raises(HTTPException) as expired:
                await read_conversation(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    settings,
                    credential,
                )
            assert expired.value.status_code == 410
            row = await db.get(ConversationRecord, conversation_id)
            assert row is not None
            row.idle_expires_at = datetime.now(UTC) + timedelta(minutes=5)
        terminal = next_state.model_copy(
            update={"turn": 2, "status": "completed", "slot_values": ()}
        )
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            lease = await reserve_turn(
                db,
                await user(db, user_id),
                org_id,
                conversation_id,
                2,
                settings,
                credential,
            )
            assert (
                await commit_turn(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    2,
                    lease.token,
                    terminal,
                    settings,
                    credential,
                )
            ).revision == 3
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=user_id
        ) as db:
            row = await db.scalar(
                select(ConversationRecord).where(ConversationRecord.id == conversation_id)
            )
            assert row is not None and row.checkpoint is None
            with pytest.raises(HTTPException) as finished:
                await read_conversation(
                    db,
                    await user(db, user_id),
                    org_id,
                    conversation_id,
                    settings,
                    credential,
                )
            assert finished.value.status_code == 409
        from dialogentis.runtime.cleanup import recover_abandoned_turns

        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=other_id
        ) as db:
            abandoned = await reserve_turn(
                db, await user(db, other_id), org_id, own.id, 1, settings, own_credential
            )
            assert str(abandoned.token) not in repr(abandoned)
            with pytest.raises(HTTPException) as busy:
                await read_conversation(
                    db, await user(db, other_id), org_id, own.id, settings, own_credential
                )
            assert busy.value.status_code == 409 and busy.value.headers == {"Retry-After": "1"}
            row = await db.get(ConversationRecord, own.id)
            assert row is not None
            row.processing_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=other_id
        ) as db:
            with pytest.raises(HTTPException) as expired_lease:
                await commit_turn(
                    db,
                    await user(db, other_id),
                    org_id,
                    own.id,
                    1,
                    abandoned.token,
                    next_state,
                    settings,
                    own_credential,
                )
            assert expired_lease.value.status_code == 409
            assert await recover_abandoned_turns(db, org_id) == 1
            assert await recover_abandoned_turns(db, org_id) == 0
        async with unit_of_work(
            app.state.session_factory, organization_id=org_id, user_id=other_id
        ) as db:
            row = await db.get(ConversationRecord, own.id)
            assert (
                row is not None
                and row.checkpoint is None
                and row.processing_token is None
                and row.revision == 2
            )
            with pytest.raises(HTTPException) as recovered:
                await commit_turn(
                    db,
                    await user(db, other_id),
                    org_id,
                    own.id,
                    1,
                    abandoned.token,
                    next_state,
                    settings,
                    own_credential,
                )
            assert recovered.value.status_code == 409
