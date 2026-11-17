import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError, IntegrityError

from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.integrations.ledger import begin_action, finish_action
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import User
from dialogentis.runtime.cleanup import cleanup_expired_conversations, recover_abandoned_turns
from dialogentis.runtime.core import RuntimeResult
from dialogentis.runtime.idempotency import ReservedMessage, complete_message, reserve_message
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


async def test_action_ledger_is_fenced_and_recovers_unknown_outcome(app):
    async with client_for(app) as owner, client_for(app) as outsider:
        user_id, org_id, agent_id = await create_agent(app, owner)
        _, other_org, _ = await create_agent(app, outsider)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "ledger-model")).version
        await process_job(app.state.session_factory, version.job_id, org_id, HANDLERS, "training")
        create = await owner.post(
            f"/api/v1/organizations/{org_id}/integrations",
            json={
                "name": "Orders",
                "base_url": "https://example.com",
                "methods": ["GET", "POST"],
                "paths": ["/orders/"],
                "secret": "secret-token",
            },
        )
        assert create.status_code == 201, create.text
        integration_id = UUID(create.json()["id"])
        service = app.state.test_chat_service
        chat = await service.start(user_id, org_id, agent_id, version.id)
        assert chat.credential is not None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            turn = await reserve_message(
                db,
                user,
                org_id,
                chat.conversation_id,
                chat.credential,
                1,
                uuid4(),
                "ledger-message",
                "private-message",
                app.state.settings,
            )
            assert isinstance(turn, ReservedMessage)
        node = uuid4()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            first = await begin_action(
                db,
                user,
                org_id,
                chat.conversation_id,
                chat.credential,
                turn,
                integration_id,
                node,
                1,
                "POST",
            )
            assert str(first.token) not in repr(first)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            with pytest.raises(HTTPException) as duplicate:
                await begin_action(
                    db,
                    user,
                    org_id,
                    chat.conversation_id,
                    chat.credential,
                    turn,
                    integration_id,
                    node,
                    1,
                    "POST",
                )
            assert duplicate.value.status_code == 409
            with pytest.raises(HTTPException) as parallel:
                await begin_action(
                    db,
                    user,
                    org_id,
                    chat.conversation_id,
                    chat.credential,
                    turn,
                    integration_id,
                    node,
                    2,
                    "POST",
                )
            assert parallel.value.status_code == 409
            with pytest.raises(HTTPException) as incomplete:
                await complete_message(
                    db,
                    user,
                    org_id,
                    chat.conversation_id,
                    chat.credential,
                    turn,
                    RuntimeResult(turn.reservation.state, (), ()),
                    app.state.settings,
                )
            assert incomplete.value.status_code == 409
            stale = replace(turn, reservation=replace(turn.reservation, token=uuid4()))
            with pytest.raises(HTTPException):
                await finish_action(
                    db, user, org_id, chat.credential, stale, first, "succeeded", 200
                )
            await finish_action(db, user, org_id, chat.credential, turn, first, "succeeded", 200)
            with pytest.raises(HTTPException):
                await finish_action(
                    db, user, org_id, chat.credential, turn, first, "succeeded", 200
                )
            second = await begin_action(
                db,
                user,
                org_id,
                chat.conversation_id,
                chat.credential,
                turn,
                integration_id,
                node,
                2,
                "GET",
            )
            await finish_action(db, user, org_id, chat.credential, turn, second, "http_error")
            third = await begin_action(
                db,
                user,
                org_id,
                chat.conversation_id,
                chat.credential,
                turn,
                integration_id,
                node,
                3,
                "POST",
            )
            with pytest.raises(ValueError):
                await begin_action(
                    db,
                    user,
                    org_id,
                    chat.conversation_id,
                    chat.credential,
                    turn,
                    integration_id,
                    node,
                    4,
                    "POST",
                )
        with pytest.raises(IntegrityError):
            async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
                assert await db.get(ActionExecution, first.id) is None
                db.add(
                    ActionExecution(
                        id=uuid4(),
                        organization_id=other_org,
                        conversation_id=chat.conversation_id,
                        receipt_id=turn.turn_id,
                        integration_id=integration_id,
                        node_id=node,
                        processing_token=turn.reservation.token,
                        ordinal=1,
                        method="GET",
                        status="started",
                    )
                )
                await db.flush()
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            with pytest.raises(DBAPIError):
                async with dispatcher() as db:
                    await db.execute(select(ActionExecution))
        finally:
            await dispatcher.kw["bind"].dispose()
        item_path = f"/api/v1/organizations/{org_id}/integrations/{integration_id}"
        assert (
            await owner.delete(item_path, headers={"If-Match": create.headers["etag"]})
        ).status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            conversation = await db.get(ConversationRecord, chat.conversation_id)
            assert conversation is not None
            conversation.processing_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await recover_abandoned_turns(db, org_id) == 1
            action = await db.get(ActionExecution, third.id)
            receipt = await db.get(RuntimeReceipt, turn.turn_id)
            assert (
                action is not None
                and action.status == "unknown_result"
                and action.finished_at is not None
            )
            assert receipt is not None and receipt.status == "unknown"
            user = await db.get(User, user_id)
            assert user is not None
            with pytest.raises(HTTPException):
                await finish_action(
                    db, user, org_id, chat.credential, turn, third, "succeeded", 200
                )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await db.execute(delete(RuntimeReceipt).where(RuntimeReceipt.id == turn.turn_id))
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(ActionExecution, first.id) is None
            assert await db.get(ActionExecution, second.id) is None
            assert await db.get(ActionExecution, third.id) is None
        assert (
            await owner.delete(item_path, headers={"If-Match": create.headers["etag"]})
        ).status_code == 204
        fresh = await owner.post(
            f"/api/v1/organizations/{org_id}/integrations",
            json={
                "name": "New",
                "base_url": "https://example.com",
                "methods": ["GET"],
                "paths": ["/"],
            },
        )
        assert fresh.status_code == 201
        next_integration = UUID(fresh.json()["id"])
        next_chat = await service.start(user_id, org_id, agent_id, version.id)
        assert next_chat.credential is not None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            next_turn = await reserve_message(
                db,
                user,
                org_id,
                next_chat.conversation_id,
                next_chat.credential,
                1,
                uuid4(),
                "next-ledger-message",
                "text",
                app.state.settings,
            )
            assert isinstance(next_turn, ReservedMessage)
            final_action = await begin_action(
                db,
                user,
                org_id,
                next_chat.conversation_id,
                next_chat.credential,
                next_turn,
                next_integration,
                uuid4(),
                1,
                "GET",
            )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, next_chat.conversation_id)
            assert row is not None
            row.idle_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await cleanup_expired_conversations(db, org_id) == 1
            pending = await db.get(ActionExecution, final_action.id)
            receipt = await db.get(RuntimeReceipt, next_turn.turn_id)
            assert pending is not None and pending.status == "unknown_result"
            assert receipt is not None and receipt.status == "unknown"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(ActionExecution, final_action.id) is None
