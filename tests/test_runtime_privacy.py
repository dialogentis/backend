import asyncio
import json
import os
from uuid import UUID, uuid4

import pytest
from aio_pika import ExchangeType, Message, connect
from sqlalchemy import Table, Text, cast, func, select

from dialogentis.db import unit_of_work
from dialogentis.jobs.models import OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.history_models import HistoryTurn
from dialogentis.runtime.models import TestConversation as ConversationRecord
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.telemetry import logger
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_nlu_exception_does_not_retain_or_publish_private_input(app, monkeypatch, caplog):
    async with client_for(app) as owner:
        user_id, org_id, agent_id = await create_agent(app, owner)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "privacy-nlu")).version
        await mark_published(app, org_id)
        factory = app.state.session_factory
        await process_job(factory, version.job_id, org_id, HANDLERS, "training")
        base = f"/api/v1/organizations/{org_id}"
        settings = await owner.get(base)
        changed = await owner.patch(
            base,
            json={"store_message_text": False},
            headers={"If-Match": settings.headers["etag"]},
        )
        assert changed.status_code == 200
        started = await owner.post(
            f"{base}/agents/{agent_id}/test-conversations",
            json={"version_id": str(version.id)},
        )
        assert started.status_code == 201, started.text
        conversation = started.json()
        service = app.state.test_chat_service
        await service.executor.close()
        service.executor = RuntimeExecutor()
        identifier = UUID(conversation["conversation_id"])
        async with unit_of_work(factory, organization_id=org_id) as db:
            before = set(await db.scalars(select(OutboxEvent.id)))

        program = """
from dialogentis.runtime import child

def predictor(_):
    def fail(message):
        raise RuntimeError(message.text)
    return fail

child.nlu_predictor = predictor
child.main()
"""
        spawn = asyncio.create_subprocess_exec

        async def failing_model(executable, *args, **kwargs):
            assert args == ("-m", "dialogentis.runtime.child")
            return await spawn(executable, "-c", program, **kwargs)

        marker = f"private-nlu-{uuid4().hex}"
        broker_url = app.state.settings.broker_url
        assert broker_url is not None
        records = []
        async with await connect(broker_url.get_secret_value(), timeout=5) as connection:
            channel = await connection.channel()
            exchange = await channel.declare_exchange(
                "dialogentis", ExchangeType.TOPIC, durable=True
            )
            probe = await channel.declare_queue(exclusive=True)
            await probe.bind(exchange, routing_key="#")
            await exchange.publish(Message(b"privacy-probe"), routing_key=f"test.{uuid4().hex}")
            echo = await probe.get(timeout=5)
            assert echo is not None and echo.body == b"privacy-probe"
            await echo.ack()
            with monkeypatch.context() as patch, caplog.at_level("DEBUG"):
                patch.setattr(
                    logger, "log", lambda _level, record: records.append(json.loads(record))
                )
                patch.setattr(
                    "dialogentis.runtime.executor.asyncio.create_subprocess_exec", failing_model
                )
                result = await owner.post(
                    f"{base}/test-conversations/{identifier}/messages",
                    json={"client_message_id": str(uuid4()), "revision": 1, "text": marker},
                    headers={
                        "X-Conversation-Token": conversation["conversation_token"],
                        "Idempotency-Key": "nlu-error",
                    },
                )
            assert await probe.get(fail=False, timeout=5) is None
        assert result.status_code == 200 and result.json()["error_code"] == "nlu_error"
        assert (
            marker not in result.text
            and marker not in caplog.text
            and marker not in json.dumps(records)
        )
        request_id = result.headers["x-request-id"]
        turn_id = result.json()["turn_id"]
        assert any(
            entry.get("request_id") == request_id
            and entry.get("turn_id") == turn_id
            and entry.get("error_code") == "nlu_error"
            and entry["event"] == "turn_finished"
            for entry in records
        )
        async with unit_of_work(factory, organization_id=org_id) as db:
            identifier_turn = UUID(result.json()["turn_id"])
            receipt = await db.get(RuntimeReceipt, identifier_turn)
            assert receipt is not None and receipt.encrypted_response is None
            history = await db.get(HistoryTurn, (org_id, identifier, 1))
            assert history is not None and history.encrypted_messages is None
            row = await db.get(ConversationRecord, identifier)
            assert row is not None and row.checkpoint is None
            assert set(await db.scalars(select(OutboxEvent.id))) == before
            for model in (HistoryTurn, RuntimeReceipt, ConversationRecord, OutboxEvent):
                table = model.__table__
                assert isinstance(table, Table)
                content = cast(func.row_to_json(table.table_valued()), Text)
                assert (
                    await db.scalar(
                        select(func.count()).select_from(model).where(content.contains(marker))
                    )
                    == 0
                )
        for path in (app.state.settings.artifact_root / str(org_id)).iterdir():
            if path.is_file():
                assert marker.encode() not in path.read_bytes()
