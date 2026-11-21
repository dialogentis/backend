import os
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError

from dialogentis.analytics import cleanup_metrics, record_accepted, record_completed
from dialogentis.analytics_models import MetricEvent, MetricHour
from dialogentis.db import unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import AgentDraft, User
from dialogentis.runtime.cleanup import recover_abandoned_turns
from dialogentis.runtime.core import RuntimeResult, RuntimeState
from dialogentis.runtime.history_models import ConversationHistory
from dialogentis.runtime.idempotency import ReservedMessage, reserve_message
from dialogentis.runtime.models import TestConversation as ConversationRecord
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_runtime_core import edge, node
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_runtime_counts_replay_handoff_and_abandoned_turn_once(app):
    async with client_for(app) as owner, client_for(app) as outsider:
        user_id, org_id, root_id = await create_agent(app, owner)
        _, other_org, _ = await create_agent(app, outsider)
        base = f"/api/v1/organizations/{org_id}"
        created = await owner.post(f"{base}/agents", json={"name": "Target", "slug": "target"})
        assert created.status_code == 201
        target_id = UUID(created.json()["id"])
        factory = app.state.session_factory
        await populate(app, org_id, target_id)
        await populate(app, org_id, root_id)
        async with unit_of_work(factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == target_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            intent_id = config["intents"][0]["id"]
            switch = config["flow"]["nodes"][1]
            returned = node("return")
            config["flow"]["nodes"].append(returned)
            config["flow"]["edges"].append(edge(switch, returned, intent_id))
            draft.configuration = config
        target = (await prepare(app, user_id, org_id, target_id, 1, "analytics-target")).version
        await process_job(factory, target.job_id, org_id, HANDLERS, "training")
        async with unit_of_work(factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == root_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            switch, end = config["flow"]["nodes"][1], config["flow"]["nodes"][3]
            handoff = node(
                "handoff",
                target_agent_id=str(target_id),
                target_version_id=str(target.id),
                forward_current_message=True,
            )
            config["flow"]["nodes"].append(handoff)
            config["flow"]["edges"] += [
                edge(switch, handoff, intent_id),
                edge(handoff, switch, "returned"),
                edge(handoff, end, "error"),
            ]
            draft.configuration = config
        root = (await prepare(app, user_id, org_id, root_id, 1, "analytics-root")).version
        await process_job(factory, root.job_id, org_id, HANDLERS, "training")
        await mark_published(app, org_id)
        start_url = f"{base}/agents/{root_id}/test-conversations"
        start_headers = {"Idempotency-Key": "one-analytics-dialogue"}
        start_body = {"version_id": str(root.id)}
        started = await owner.post(start_url, json=start_body, headers=start_headers)
        assert started.status_code == 201, started.text
        assert (await owner.post(start_url, json=start_body, headers=start_headers)).json()[
            "replayed"
        ]
        first = started.json()
        conversation = UUID(first["conversation_id"])
        endpoint = f"{base}/test-conversations/{conversation}/messages"
        body = {"client_message_id": str(uuid4()), "revision": 1, "text": "где заказ 0"}
        token = first["conversation_token"]
        headers = {"X-Conversation-Token": token, "Idempotency-Key": "one-message"}
        result = await owner.post(endpoint, json=body, headers=headers)
        assert result.status_code == 200, result.text
        assert any(event["event"] == "returned" for event in result.json()["trace"])
        assert (await owner.post(endpoint, json=body, headers=headers)).json()["replayed"]
        fallback = await owner.post(
            endpoint,
            json={"client_message_id": str(uuid4()), "revision": 2, "text": "🛸🛸"},
            headers={**headers, "Idempotency-Key": "fallback-message"},
        )
        assert fallback.status_code == 200
        assert any(event["event"] == "fallback" for event in fallback.json()["trace"])
        async with unit_of_work(factory, organization_id=org_id) as db:
            user = await db.get(User, user_id)
            assert user is not None
            reserved = await reserve_message(
                db,
                user,
                org_id,
                conversation,
                token,
                3,
                uuid4(),
                "abandoned",
                "Контрольная строка",
                app.state.settings,
            )
            assert isinstance(reserved, ReservedMessage)
            row = await db.get(ConversationRecord, conversation)
            assert row is not None
            row.processing_expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await db.flush()
            assert await recover_abandoned_turns(db, org_id) == 1
            assert await recover_abandoned_turns(db, org_id) == 0
            await db.execute(
                delete(ConversationHistory).where(ConversationHistory.id == conversation)
            )
        async with unit_of_work(factory, organization_id=org_id) as db:
            rows = (await db.scalars(select(MetricHour))).all()
            totals: Counter[str] = Counter()
            for row in rows:
                totals[row.metric] += row.count
                assert row.count == sum(row.minute_counts.values())
                assert row.channel == "test" and row.root_agent_id == root_id
            assert totals == {
                "dialogues": 1,
                "messages": 3,
                "completed": 3,
                "failed": 1,
                "nlu": 2,
                "fallback": 1,
                "latency_total": 2,
                "latency_internal": 2,
                "latency_http": 2,
                "intent": 2,
            }
            assert {
                (row.active_agent_id, row.item, row.count) for row in rows if row.metric == "intent"
            } == {
                (root_id, intent_id, 1),
                (target_id, intent_id, 1),
            }
        async with unit_of_work(factory, organization_id=other_org) as db:
            assert (await db.scalars(select(MetricHour))).first() is None
            assert (await db.scalars(select(MetricEvent))).first() is None
        report = f"{base}/analytics"
        assert (await outsider.get(report)).status_code == 404
        production = (await owner.get(report)).json()
        assert production["messages"] == 0 and production["series"] == []
        assert production["fallback"] == {"count": 0, "total": 0, "value": None}
        response = await owner.get(report, params={"channel": "test"})
        assert response.status_code == 200, response.text
        test = response.json()
        assert (test["dialogues"], test["messages"]) == (1, 3)
        assert test["fallback"] == {"count": 1, "total": 2, "value": 0.5}
        assert test["errors"] == {"count": 1, "total": 3, "value": 1 / 3}
        assert test["latency"]["samples"] == 2
        assert [(item["agent_id"], item["intent_name"]) for item in test["intents"]] == [
            (str(root_id), "Заказ"),
        ]
        active = await owner.get(
            report, params={"channel": "test", "agent_id": str(target_id), "intent_scope": "active"}
        )
        assert active.status_code == 200, active.text
        assert active.json()["messages"] == 0
        assert [(item["agent_id"], item["intent_name"]) for item in active.json()["intents"]] == [
            (str(target_id), "Заказ"),
        ]
        filters = {"channel_type": "test", "timezone": "UTC"}
        summary = await owner.get(f"{report}/summary", params=filters)
        assert summary.status_code == 200 and summary.json()["messages"] == 3
        for section, field in (
            ("timeseries", "series"),
            ("intents", "intents"),
            ("errors", "errors"),
        ):
            response = await owner.get(f"{report}/{section}", params=filters)
            assert response.status_code == 200 and response.json()[field] == test[field]


async def test_hourly_upsert_preserves_minutes_and_expires_deduplication_keys(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        factory = app.state.session_factory
        row = ConversationRecord(organization_id=org_id, agent_id=agent_id, version_id=uuid4())
        first, second, duplicate = uuid4(), uuid4(), uuid4()
        start = datetime(2026, 11, 19, 18, tzinfo=UTC)
        async with unit_of_work(factory, organization_id=org_id) as db:
            for turn_id, minute in [(first, 5), (second, 35), (duplicate, 35), (duplicate, 36)]:
                await record_accepted(
                    db, row, turn_id, start + timedelta(minutes=minute), kind="messages"
                )
            totals = (await db.scalars(select(MetricHour))).one()
            assert totals.bucket_at == start and totals.count == 3
            assert totals.minute_counts == {"5": 1, "35": 2}
            assert len((await db.scalars(select(MetricEvent))).all()) == 3
            state = RuntimeState(
                version_id=row.version_id,
                flow_hash="f" * 64,
                node_id=uuid4(),
                turn=1,
                status="completed",
                fallback_count=0,
            )
            result = RuntimeResult(state, (), ())
            for turn_id, minute, milliseconds in [
                (first, 5, 40),
                (second, 35, 30),
                (duplicate, 35, 180),
            ]:
                accepted_at = start + timedelta(minutes=minute)
                await record_completed(
                    db,
                    row,
                    turn_id,
                    accepted_at,
                    accepted_at + timedelta(milliseconds=milliseconds),
                    result,
                )
        report = f"/api/v1/organizations/{org_id}/analytics"
        boundary = {"channel": "test", "time_zone": "Asia/Kathmandu"}
        early = await owner.get(
            report, params={**boundary, "from": "2026-11-19", "to": "2026-11-19"}
        )
        later = await owner.get(
            report, params={**boundary, "from": "2026-11-20", "to": "2026-11-20"}
        )
        assert early.status_code == later.status_code == 200, later.text
        assert early.json()["messages"] == 1 and later.json()["messages"] == 2
        assert later.json()["series"] == [{"day": "2026-11-20", "dialogues": 0, "messages": 2}]
        assert later.json()["latency"] == {"samples": 2, "p50_ms": 50, "p95_ms": 250}
        assert later.json()["external_http_latency"]["p95_ms"] == 0
        assert (await owner.get(report, params={"time_zone": "Europe/Invalid"})).status_code == 422
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            async with dispatcher() as db:
                assert (
                    org_id
                    in (await db.scalars(select(MetricHour.organization_id).distinct())).all()
                )
                with pytest.raises(DBAPIError):
                    await db.execute(select(MetricHour.count))
        finally:
            await dispatcher.kw["bind"].dispose()
        async with unit_of_work(factory, organization_id=org_id) as db:
            await cleanup_metrics(db, org_id, start + timedelta(days=91))
            assert (await db.scalars(select(MetricHour))).first() is None
            assert (await db.scalars(select(MetricEvent))).first() is None
