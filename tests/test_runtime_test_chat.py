import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select

from dialogentis.db import unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import Membership, Organization
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.models import TestConversation as ConversationRecord
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.test_chat import TestChatService as ChatService
from dialogentis.runtime.watchdog import recover_once
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_real_model_chat_and_revoked_access_during_execution(app, monkeypatch):
    async with client_for(app) as owner, client_for(app) as viewer:
        user_id, org_id, agent_id = await create_agent(app, owner)
        viewer_id, _, _ = await create_agent(app, viewer)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "real-test-chat")).version
        version_id, job_id = version.id, version.job_id
        await process_job(app.state.session_factory, job_id, org_id, HANDLERS, "training")
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        executor = RuntimeExecutor()
        service = ChatService(app.state.session_factory, app.state.settings, executor)
        started = await service.start(viewer_id, org_id, agent_id, version_id)
        assert started.revision == 1 and started.status == "waiting_input"
        assert started.credential is not None and started.credential not in repr(started)
        message_id = uuid4()
        with pytest.raises(HTTPException) as bad_token:
            await service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                1,
                "🛸🛸",
                "x" * 43,
                message_id,
                "first-key",
            )
        assert bad_token.value.status_code == 404
        result = await service.send(
            viewer_id,
            org_id,
            started.conversation_id,
            1,
            "🛸🛸",
            started.credential,
            message_id,
            "first-key",
        )
        assert result.revision == 2 and "🛸🛸" not in repr(result)
        replay = await service.send(
            viewer_id,
            org_id,
            started.conversation_id,
            1,
            "🛸🛸",
            started.credential,
            message_id,
            "first-key",
        )
        assert (
            replay.replayed
            and replay.replay_available
            and replay.turn_id == result.turn_id
            and replay.messages == result.messages
        )
        with pytest.raises(HTTPException) as reused:
            await service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                1,
                "другой текст",
                started.credential,
                message_id,
                "first-key",
            )
        assert reused.value.status_code == 409
        with pytest.raises(HTTPException) as stale:
            await service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                1,
                "новый вопрос",
                started.credential,
                uuid4(),
                "stale-key",
            )
        assert stale.value.status_code == 409
        with pytest.raises(HTTPException) as wrong_user:
            await service.send(
                user_id,
                org_id,
                started.conversation_id,
                2,
                "новый вопрос",
                started.credential,
                uuid4(),
                "wrong-user",
            )
        assert wrong_user.value.status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, started.conversation_id)
            assert (
                row is not None
                and row.checkpoint is not None
                and "🛸🛸".encode() not in row.checkpoint
            )

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org = await db.get(Organization, org_id)
            assert org is not None
            org.store_message_text = False
        private = await service.start(viewer_id, org_id, agent_id, version_id)
        assert private.credential is not None
        private_message = uuid4()
        first_private = await service.send(
            viewer_id,
            org_id,
            private.conversation_id,
            1,
            "🛸🛸",
            private.credential,
            private_message,
            "private-key",
        )
        assert not first_private.replay_available and first_private.messages
        replay_private = await service.send(
            viewer_id,
            org_id,
            private.conversation_id,
            1,
            "🛸🛸",
            private.credential,
            private_message,
            "private-key",
        )
        assert (
            replay_private.replayed
            and not replay_private.replay_available
            and replay_private.messages == ()
        )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            receipt = await db.scalar(
                select(RuntimeReceipt).where(RuntimeReceipt.id == first_private.turn_id)
            )
            assert receipt is not None and receipt.encrypted_response is None

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org = await db.get(Organization, org_id)
            assert org is not None
            org.store_message_text = True
        failed_message = uuid4()

        def fail_to_seal(*args, **kwargs):
            raise RuntimeError("Simulated receipt failure")

        with monkeypatch.context() as patch:
            patch.setattr("dialogentis.runtime.idempotency.seal_receipt", fail_to_seal)
            with pytest.raises(RuntimeError, match="Simulated receipt failure"):
                await service.send(
                    viewer_id,
                    org_id,
                    private.conversation_id,
                    2,
                    "🧩🧩",
                    private.credential,
                    failed_message,
                    "failed-key",
                )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, private.conversation_id)
            failed = await db.scalar(
                select(RuntimeReceipt).where(RuntimeReceipt.client_message_id == failed_message)
            )
            assert row is not None and row.revision == 2 and row.checkpoint is not None
            assert (
                failed is not None
                and failed.status == "processing"
                and failed.encrypted_response is None
            )

        disconnected = await service.start(viewer_id, org_id, agent_id, version_id)
        assert disconnected.credential is not None
        disconnect_ready, disconnect_release = asyncio.Event(), asyncio.Event()
        real_execute = executor.execute

        async def pause_for_disconnect(**kwargs):
            disconnect_ready.set()
            await disconnect_release.wait()
            return await real_execute(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(executor, "execute", pause_for_disconnect)
            disconnected_id = uuid4()
            caller = asyncio.create_task(
                service.send(
                    viewer_id,
                    org_id,
                    disconnected.conversation_id,
                    1,
                    "🛸🛸",
                    disconnected.credential,
                    disconnected_id,
                    "disconnect-key",
                )
            )
            try:
                await asyncio.wait_for(disconnect_ready.wait(), 10)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            finally:
                disconnect_release.set()
            completed = await asyncio.wait_for(asyncio.gather(*service._pending), 20)
            assert len(completed) == 1 and completed[0].messages
        after_disconnect = await service.send(
            viewer_id,
            org_id,
            disconnected.conversation_id,
            1,
            "🛸🛸",
            disconnected.credential,
            disconnected_id,
            "disconnect-key",
        )
        assert after_disconnect.replayed and after_disconnect.messages == completed[0].messages

        ready, release = asyncio.Event(), asyncio.Event()
        real_execute = executor.execute

        async def paused(**kwargs):
            ready.set()
            await release.wait()
            return await real_execute(**kwargs)

        monkeypatch.setattr(executor, "execute", paused)
        pending_id = uuid4()
        task = asyncio.create_task(
            service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                2,
                "🧩🧩",
                started.credential,
                pending_id,
                "revoked-key",
            )
        )
        await asyncio.wait_for(ready.wait(), 10)
        with pytest.raises(HTTPException) as busy:
            await service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                2,
                "🧩🧩",
                started.credential,
                pending_id,
                "revoked-key",
            )
        assert busy.value.status_code == 409
        try:
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                await db.execute(
                    delete(Membership).where(
                        Membership.organization_id == org_id, Membership.user_id == viewer_id
                    )
                )
        finally:
            release.set()
        with pytest.raises(HTTPException) as revoked:
            await task
        assert revoked.value.status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, started.conversation_id)
            assert row is not None and row.revision == 2 and row.processing_token is not None
            receipt = await db.scalar(
                select(RuntimeReceipt).where(RuntimeReceipt.client_message_id == pending_id)
            )
            assert receipt is not None and receipt.status == "processing"
            row.processing_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            assert await recover_once(app.state.session_factory, dispatcher) >= 1
            assert await recover_once(app.state.session_factory, dispatcher) == 0
        finally:
            await dispatcher.kw["bind"].dispose()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            receipt = await db.scalar(
                select(RuntimeReceipt).where(RuntimeReceipt.client_message_id == pending_id)
            )
            assert (
                receipt is not None
                and receipt.status == "unknown"
                and receipt.encrypted_response is None
            )
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        with pytest.raises(HTTPException) as unknown:
            await service.send(
                viewer_id,
                org_id,
                started.conversation_id,
                2,
                "🧩🧩",
                started.credential,
                pending_id,
                "revoked-key",
            )
        assert unknown.value.status_code == 409 and "неизвестен" in unknown.value.detail
