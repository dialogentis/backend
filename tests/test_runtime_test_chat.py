import asyncio
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select

from dialogentis.db import unit_of_work
from dialogentis.http_action import HTTP_OUTPUTS
from dialogentis.integrations.http_client import IntegrationRequestError, IntegrationResponse
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import AgentDraft, Membership
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


async def test_mock_and_live_http_actions_use_organization_policy(app, monkeypatch):
    async with client_for(app) as owner, client_for(app) as viewer:
        user_id, org_id, agent_id = await create_agent(app, owner)
        viewer_id, _, _ = await create_agent(app, viewer)
        await populate(app, org_id, agent_id)
        integrations = f"/api/v1/organizations/{org_id}/integrations"
        created = await owner.post(
            integrations,
            json={
                "name": "Orders",
                "base_url": "https://orders.example.com",
                "methods": ["POST"],
                "paths": ["/orders/"],
                "response_fields": ["order.status"],
                "secret": "private-token",
                "supports_idempotency": True,
            },
        )
        assert created.status_code == 201
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            slot_id, action_id, say_id = (str(uuid4()) for _ in range(3))
            config["slots"] = [{"id": slot_id, "code": "status", "sensitive": True}]
            action = {
                "id": action_id,
                "type": "http_action",
                "position": {"x": 0, "y": 0},
                "config": {
                    "integration_id": created.json()["id"],
                    "path": "/orders/123",
                    "method": "POST",
                    "response_mapping": {"order.status": slot_id},
                    "mock_response": {"order": {"status": "готов"}},
                },
            }
            say = {
                "id": say_id,
                "type": "say",
                "position": {"x": 0, "y": 0},
                "config": {"text": "Статус: {{status}}"},
            }
            flow = config["flow"]
            flow["nodes"] += [action, say]
            flow["edges"][0]["target"] = action_id
            end_id, switch_id = flow["nodes"][3]["id"], flow["nodes"][1]["id"]
            for edge in flow["edges"]:
                if edge["source"] == switch_id and edge["source_handle"] == "fallback":
                    edge["target"] = action_id
            flow["edges"] += [
                {
                    "id": str(uuid4()),
                    "source": action_id,
                    "source_handle": output,
                    "target": say_id if output == "success" else end_id,
                }
                for output in HTTP_OUTPUTS
            ]
            flow["edges"].append(
                {"id": str(uuid4()), "source": say_id, "source_handle": "next", "target": switch_id}
            )
            draft.configuration = config
        version = (await prepare(app, user_id, org_id, agent_id, 1, "http-mock-test")).version
        await process_job(app.state.session_factory, version.job_id, org_id, HANDLERS, "training")
        service = ChatService(app.state.session_factory, app.state.settings, RuntimeExecutor())
        result = await service.start(user_id, org_id, agent_id, version.id)
        assert result.status == "waiting_input" and result.messages == ("Статус: готов",)
        assert "готов" not in repr(result.trace) and "private-token" not in repr(result)
        executions = []

        async def transport(policy, method, path, query, body, **kwargs):
            assert method == "POST" and path == "/orders/123"
            assert kwargs["secret"] == "private-token" and kwargs["supports_idempotency"]
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                entry = await db.get(ActionExecution, kwargs["execution_id"])
                assert entry is not None and entry.status == "started"
            executions.append(kwargs["execution_id"])
            if len(executions) == 2:
                raise IntegrationRequestError("unknown_result")
            return IntegrationResponse(200, {"order": {"status": "отправлен"}, "hidden": "данные"})

        monkeypatch.setattr("dialogentis.runtime.live_http.request_json", transport)
        start_url = f"/api/v1/organizations/{org_id}/agents/{agent_id}/test-conversations"
        body = {"version_id": str(version.id), "http_mode": "live"}
        key = {"Idempotency-Key": "live-chat-start"}
        assert (await viewer.post(start_url, json=body, headers=key)).status_code == 403
        started = await owner.post(start_url, json=body, headers=key)
        assert started.status_code == 201, started.text
        live = started.json()
        assert live["messages"] == ["Статус: отправлен"]
        assert "private-token" not in started.text and "данные" not in started.text
        repeated = await owner.post(start_url, json=body, headers=key)
        assert repeated.json() == {**live, "replayed": True}
        assert (
            await owner.post(start_url, json={**body, "http_mode": "mock"}, headers=key)
        ).status_code == 409
        message_id = uuid4()
        first_message = await service.send(
            user_id,
            org_id,
            UUID(live["conversation_id"]),
            1,
            "🛸🛸",
            live["conversation_token"],
            message_id,
            "live-message",
        )
        assert any(event.output == "unknown_result" for event in first_message.trace)
        replayed = await service.send(
            user_id,
            org_id,
            UUID(live["conversation_id"]),
            1,
            "🛸🛸",
            live["conversation_token"],
            message_id,
            "live-message",
        )
        assert replayed.replayed and len(executions) == 2
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            ledger = [await db.get(ActionExecution, ident) for ident in executions]
            assert [entry.status for entry in ledger if entry is not None] == [
                "succeeded",
                "unknown_result",
            ]
        disabled = await owner.patch(
            f"{integrations}/{created.json()['id']}",
            json={"enabled": False},
            headers={"If-Match": created.headers["etag"]},
        )
        assert disabled.status_code == 200
        denied = await service.start(user_id, org_id, agent_id, version.id)
        assert denied.status == "completed" and any(
            event.output == "http_error" for event in denied.trace
        )


async def test_start_reports_errors_in_the_selected_version(app):
    async with client_for(app) as owner, client_for(app) as stranger:
        user_id, org_id, agent_id = await create_agent(app, owner)
        await create_agent(app, stranger)
        await populate(app, org_id, agent_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            configuration = deepcopy(draft.configuration)
            configuration["flow"]["edges"] = []
            draft.configuration = configuration
        version = (await prepare(app, user_id, org_id, agent_id, 1, "invalid-flow-test")).version
        await process_job(app.state.session_factory, version.job_id, org_id, HANDLERS, "training")
        path = f"/api/v1/organizations/{org_id}/agents/{agent_id}/test-conversations"
        body = {"version_id": str(version.id), "http_mode": "mock"}
        assert (await stranger.post(path, json=body)).status_code == 404
        response = await owner.post(path, json=body)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_version_flow"
        assert any(item["code"] == "missing_output" for item in error["details"])
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert not (await db.scalars(select(ConversationRecord))).all()


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

        base = f"/api/v1/organizations/{org_id}"
        chat_path = f"{base}/agents/{agent_id}/test-conversations"
        body = {"version_id": str(version_id), "http_mode": "mock"}
        assert (
            await viewer.post(chat_path, json=body, headers={"X-CSRF-Token": "bad"})
        ).status_code == 403
        assert (await viewer.post(chat_path, json={**body, "http_mode": "live"})).status_code == 422
        start_headers = {"Idempotency-Key": "http-start"}
        http_ready, http_release = asyncio.Event(), asyncio.Event()
        api_executor = app.state.test_chat_service.executor
        execute_http = api_executor.execute

        async def pause_start(**kwargs):
            http_ready.set()
            await http_release.wait()
            return await execute_http(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(api_executor, "execute", pause_start)
            start_task = asyncio.create_task(
                viewer.post(chat_path, json=body, headers=start_headers)
            )
            try:
                await asyncio.wait_for(http_ready.wait(), 10)
                in_progress = await asyncio.wait_for(
                    viewer.post(chat_path, json=body, headers=start_headers), 3
                )
                assert in_progress.status_code == 409
                assert in_progress.headers["Retry-After"] == "1"
            finally:
                http_release.set()
                created = await start_task
        assert created.status_code == 201, created.text
        chat = created.json()
        assert chat["status"] == "waiting_input" and chat["revision"] == 1
        replay_start = await viewer.post(chat_path, json=body, headers=start_headers)
        assert replay_start.status_code == 201
        assert replay_start.json() == {**chat, "replayed": True}
        message_path = f"{base}/test-conversations/{chat['conversation_id']}/messages"
        headers = {
            "X-Conversation-Token": chat["conversation_token"],
            "Idempotency-Key": "http-message",
        }
        message = {"client_message_id": str(uuid4()), "revision": 1, "text": "🛸🛸"}
        assert (await owner.post(message_path, json=message, headers=headers)).status_code == 404
        assert (
            await viewer.post(
                message_path, json=message, headers={**headers, "X-Conversation-Token": "x" * 43}
            )
        ).status_code == 404
        response = await viewer.post(message_path, json=message, headers=headers)
        assert response.status_code == 200, response.text
        replay_response = await viewer.post(message_path, json=message, headers=headers)
        assert replay_response.status_code == 200
        assert replay_response.json()["replayed"]
        assert replay_response.json()["messages"] == response.json()["messages"]
        assert (
            await viewer.post(message_path, json={**message, "text": "другое"}, headers=headers)
        ).status_code == 409

        http_ready, http_release = asyncio.Event(), asyncio.Event()
        api_executor = app.state.test_chat_service.executor
        execute_http = api_executor.execute

        async def pause_http(**kwargs):
            http_ready.set()
            await http_release.wait()
            return await execute_http(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(api_executor, "execute", pause_http)
            next_message = {"client_message_id": str(uuid4()), "revision": 2, "text": "🧩🧩"}
            next_headers = {**headers, "Idempotency-Key": "next-http-message"}
            first_http = asyncio.create_task(
                viewer.post(message_path, json=next_message, headers=next_headers)
            )
            try:
                await asyncio.wait_for(http_ready.wait(), 10)
                concurrent = await asyncio.wait_for(
                    viewer.post(
                        message_path,
                        json={**next_message, "client_message_id": str(uuid4())},
                        headers={**headers, "Idempotency-Key": "concurrent-http-message"},
                    ),
                    3,
                )
                assert concurrent.status_code == 409
                assert concurrent.headers["Retry-After"] == "1"
            finally:
                http_release.set()
            assert (await first_http).status_code == 200

        settings_url = f"/api/v1/organizations/{org_id}"
        settings = await owner.get(settings_url)
        etag = settings.headers["etag"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            aged = await db.get(RuntimeReceipt, UUID(response.json()["turn_id"]))
            assert aged is not None and aged.encrypted_response is not None
            aged.created_at = datetime.now(UTC) - timedelta(days=2)
        shorter = await owner.patch(
            settings_url, json={"history_retention_days": 1}, headers={"If-Match": etag}
        )
        assert shorter.status_code == 200 and shorter.json()["history_retention_days"] == 1
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            aged = await db.get(RuntimeReceipt, UUID(response.json()["turn_id"]))
            assert aged is not None and aged.encrypted_response is None
        etag = shorter.headers["etag"]
        change = {"store_message_text": False, "purge_existing_text": True}
        assert (
            await viewer.patch(settings_url, json=change, headers={"If-Match": etag})
        ).status_code == 403
        assert (await owner.patch(settings_url, json=change)).status_code == 428
        updated = await owner.patch(settings_url, json=change, headers={"If-Match": etag})
        assert updated.status_code == 200 and not updated.json()["store_message_text"]
        assert (
            await owner.patch(settings_url, json=change, headers={"If-Match": etag})
        ).status_code == 412
        hidden = await service.send(
            viewer_id,
            org_id,
            started.conversation_id,
            1,
            "🛸🛸",
            started.credential,
            message_id,
            "first-key",
        )
        assert hidden.replayed and not hidden.replay_available and not hidden.messages
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            original = await db.get(RuntimeReceipt, result.turn_id)
            assert original is not None and original.encrypted_response is None
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

        restored = await owner.patch(
            settings_url,
            json={"store_message_text": True},
            headers={"If-Match": updated.headers["etag"]},
        )
        assert restored.status_code == 200 and restored.json()["store_message_text"]
        still_hidden = await service.send(
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
            still_hidden.replayed
            and not still_hidden.replay_available
            and not still_hidden.messages
        )
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
