import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from aio_pika import connect

from dialogentis.db import unit_of_work
from dialogentis.jobs.models import JOB_QUEUE_NAMES, Job, OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.models import Membership
from dialogentis.telemetry import logger
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_queue_reports_outbox_workers_failures_and_tenant_visibility(app, monkeypatch):
    async with client_for(app) as owner, client_for(app) as guest:
        _, org_id, _ = await create_agent(app, owner)
        guest_id, second_org, _ = await create_agent(app, guest)
        url = f"/api/v1/organizations/{org_id}/jobs/queue"
        assert (await guest.get(url)).status_code == 404
        now = datetime.now(UTC)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=guest_id, role="viewer"))
            dispatch = Job(organization_id=org_id, kind="history.deletion", status="queued")
            worker = Job(
                organization_id=org_id, kind="agent.training", queue="training", status="queued"
            )
            retry = Job(
                organization_id=org_id,
                kind="dataset.export",
                status="queued",
                next_retry_at=now + timedelta(minutes=3),
            )
            failed = Job(
                organization_id=org_id,
                kind="dataset.import",
                status="failed",
                error_code="worker_lost",
            )
            db.add_all([dispatch, worker, retry, failed])
            await db.flush()
            ids = [str(row.id) for row in (dispatch, worker, retry, failed)]
            db.add_all(
                [
                    OutboxEvent(organization_id=org_id, job_id=dispatch.id),
                    OutboxEvent(
                        organization_id=org_id, job_id=worker.id, queue="training", published_at=now
                    ),
                    OutboxEvent(
                        organization_id=org_id, job_id=retry.id, available_at=retry.next_retry_at
                    ),
                ]
            )
        async with unit_of_work(app.state.session_factory, organization_id=second_org) as db:
            foreign = Job(
                organization_id=second_org, kind="agent.training", queue="training", status="failed"
            )
            db.add(foreign)
            await db.flush()
            foreign_id = str(foreign.id)

        settings = app.state.settings
        assert settings.broker_url is not None
        isolated = f"test.workspace.{uuid4()}"
        monkeypatch.setattr(
            "dialogentis.jobs.overview.JOB_QUEUE_NAMES", {**JOB_QUEUE_NAMES, "normal": isolated}
        )
        async with await connect(settings.broker_url.get_secret_value(), timeout=5) as connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(isolated)

            async def acknowledge(message):
                await message.ack()

            try:
                tag = await queue.consume(acknowledge)
                try:
                    response = await owner.get(url)
                    assert response.status_code == 200, response.text
                    report = response.json()
                    assert (
                        report["active_agents"] == 1
                        and report["queued"] == 3
                        and report["failed"] == 1
                    )
                    assert report["broker"]["status"] == "available"
                    assert report["broker"]["normal_worker"] == "listening"
                    assert {row["job_id"]: row["phase"] for row in report["items"]} == dict(
                        zip(ids, ["dispatch_wait", "worker_wait", "retry_wait", None], strict=True)
                    )
                finally:
                    await queue.cancel(tag)
                assert (await owner.get(url)).json()["broker"]["normal_worker"] == "absent"
            finally:
                await queue.delete()
        monkeypatch.setattr(app.state, "settings", settings.model_copy(update={"broker_url": None}))
        active = (await owner.get(url, params={"status": "active", "limit": 2})).json()
        assert active["broker"] == {
            "status": "unavailable",
            "normal_worker": "unknown",
            "training_worker": "unknown",
        }
        final = (
            await owner.get(url, params={"status": "active", "cursor": active["next_cursor"]})
        ).json()
        assert {item["job_id"] for item in [*active["items"], *final["items"]]} == set(ids[:3])
        assert [
            item["job_id"]
            for item in (await owner.get(url, params={"status": "failed"})).json()["items"]
        ] == [ids[3]]
        assert (await owner.get(url, params={"cursor": foreign_id})).status_code == 404
        assert [item["job_id"] for item in (await guest.get(url)).json()["items"]] == [ids[1]]


async def test_job_logs_join_api_request_and_worker_attempt(app, monkeypatch):
    async with client_for(app) as owner:
        _, org_id, _ = await create_agent(app, owner)
        entries = []
        monkeypatch.setattr(
            logger, "log", lambda _level, record: entries.append(json.loads(record))
        )
        response = await owner.post(
            f"/api/v1/organizations/{org_id}/history/exports", json={"format": "json"}
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await mark_published(app, org_id)
        await process_job(app.state.session_factory, UUID(job_id), org_id)
        joined = [item for item in entries if item.get("job_id") == job_id]
        assert [item["event"] for item in joined] == [
            "job_enqueue_requested",
            "job_claimed",
            "job_attempt_finished",
        ]
        assert joined[0]["request_id"] == response.headers["x-request-id"]
        assert "request_id" not in joined[1] and "request_id" not in joined[2]
