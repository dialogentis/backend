import os
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, update

from dialogentis.db import unit_of_work
from dialogentis.integrations.ledger import begin_action, finish_action
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.jobs.models import OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import Organization, User
from dialogentis.runtime.cleanup import recover_abandoned_turns
from dialogentis.runtime.core import Prediction, RuntimeEngine
from dialogentis.runtime.idempotency import (
    ReplayedStart,
    ReservedMessage,
    complete_initial_turn,
    reserve_initial_turn,
    reserve_message,
)
from dialogentis.runtime.models import TestConversation as ConversationRecord
from dialogentis.runtime.receipt_models import RuntimeReceipt
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_initial_turn_is_fenced_and_recovers_actions(app):
    async with client_for(app) as owner:
        user_id, org_id, agent_id = await create_agent(app, owner)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "initial-model")).version
        await process_job(app.state.session_factory, version.job_id, org_id, HANDLERS, "training")
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await db.execute(
                update(OutboxEvent)
                .where(OutboxEvent.organization_id == org_id, OutboxEvent.job_id == version.job_id)
                .values(published_at=func.now())
            )
        created = await owner.post(
            f"/api/v1/organizations/{org_id}/integrations",
            json={
                "name": "Orders",
                "base_url": "https://example.com",
                "methods": ["POST"],
                "paths": ["/orders/"],
            },
        )
        assert created.status_code == 201
        integration_id = UUID(created.json()["id"])
        engine = RuntimeEngine(
            version.id, version.snapshot["config"], lambda _: Prediction(version.id, None)
        )
        initial = engine.initial_state()
        result = engine.start()
        assert result.state.turn == 0 and result.state.status == "waiting_input"

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            reserved = await reserve_initial_turn(
                db,
                user,
                org_id,
                agent_id,
                version.id,
                initial,
                app.state.settings,
                idempotency_key="stable-start",
            )
            assert not isinstance(reserved, ReplayedStart)
            assert reserved.credential not in repr(reserved)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            with pytest.raises(HTTPException) as pending:
                await reserve_initial_turn(
                    db,
                    user,
                    org_id,
                    agent_id,
                    version.id,
                    initial,
                    app.state.settings,
                    idempotency_key="stable-start",
                )
            assert pending.value.status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            action = await begin_action(
                db,
                user,
                org_id,
                reserved.conversation_id,
                reserved.credential,
                reserved.turn,
                integration_id,
                uuid4(),
                1,
                "POST",
            )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            with pytest.raises(HTTPException):
                await complete_initial_turn(db, user, org_id, reserved, result, app.state.settings)
            await finish_action(
                db, user, org_id, reserved.credential, reserved.turn, action, "succeeded", 200
            )
            completed = await complete_initial_turn(
                db, user, org_id, reserved, result, app.state.settings
            )
            assert completed.revision == 1 and completed.status == "waiting_input"
            with pytest.raises(HTTPException):
                await complete_initial_turn(db, user, org_id, reserved, result, app.state.settings)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            replay = await reserve_initial_turn(
                db,
                user,
                org_id,
                agent_id,
                version.id,
                initial,
                app.state.settings,
                idempotency_key="stable-start",
            )
            assert isinstance(replay, ReplayedStart)
            assert replay.conversation_id == reserved.conversation_id
            assert replay.credential == reserved.credential and replay.result.replay_available
            assert replay.result.messages == result.messages
            different = uuid4()
            with pytest.raises(HTTPException) as conflict:
                await reserve_initial_turn(
                    db,
                    user,
                    org_id,
                    agent_id,
                    different,
                    initial.model_copy(update={"version_id": different}),
                    app.state.settings,
                    idempotency_key="stable-start",
                )
            assert conflict.value.status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            following = await reserve_message(
                db,
                user,
                org_id,
                reserved.conversation_id,
                reserved.credential,
                1,
                uuid4(),
                "first",
                "hello",
                app.state.settings,
            )
            assert isinstance(following, ReservedMessage) and following.reservation.state.turn == 0

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            crashed = await reserve_initial_turn(
                db, user, org_id, agent_id, version.id, initial, app.state.settings
            )
            assert not isinstance(crashed, ReplayedStart)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            started = await begin_action(
                db,
                user,
                org_id,
                crashed.conversation_id,
                crashed.credential,
                crashed.turn,
                integration_id,
                uuid4(),
                1,
                "POST",
            )
            row = await db.get(ConversationRecord, crashed.conversation_id)
            assert row is not None
            row.processing_expires_at = func.now() - timedelta(seconds=1)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await recover_abandoned_turns(db, org_id) == 1
            action_row = await db.get(ActionExecution, started.id)
            receipt = await db.get(RuntimeReceipt, crashed.turn.turn_id)
            row = await db.get(ConversationRecord, crashed.conversation_id)
            assert action_row is not None and action_row.status == "unknown_result"
            assert receipt is not None and receipt.status == "unknown"
            assert row is not None and row.checkpoint is None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org = await db.get(Organization, org_id)
            user = await db.get(User, user_id)
            assert org is not None and user is not None
            org.store_message_text = False
            private_start = await reserve_initial_turn(
                db,
                user,
                org_id,
                agent_id,
                version.id,
                initial,
                app.state.settings,
                idempotency_key="private-start",
            )
            assert not isinstance(private_start, ReplayedStart)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            await complete_initial_turn(
                db,
                user,
                org_id,
                private_start,
                replace(result, messages=("PRIVATE_SENTINEL",)),
                app.state.settings,
            )
            receipt = await db.get(RuntimeReceipt, private_start.turn.turn_id)
            assert receipt is not None and receipt.encrypted_response is None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            replay = await reserve_initial_turn(
                db,
                user,
                org_id,
                agent_id,
                version.id,
                initial,
                app.state.settings,
                idempotency_key="private-start",
            )
            assert isinstance(replay, ReplayedStart) and not replay.result.replay_available
            assert replay.credential == private_start.credential and not replay.result.messages
