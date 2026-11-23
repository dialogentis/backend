import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.lifecycle import (
    MAX_ATTEMPTS,
    MAX_DELIVERY_RECOVERIES,
    cancel,
    claim,
    complete,
    fail,
    heartbeat,
    recover_dead_letter,
    recover_tenant,
)
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.models import Organization

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


@pytest.fixture
async def factory():
    url = Settings().database_url
    assert url is not None
    value = session_factory(url.get_secret_value())
    tenants = []
    yield value, tenants
    for organization_id in tenants:
        async with unit_of_work(value, organization_id=organization_id) as db:
            await db.execute(delete(Artifact))
            await db.execute(delete(OutboxEvent))
            await db.execute(delete(Job))
            await db.execute(delete(Organization).where(Organization.id == organization_id))
    await value.kw["bind"].dispose()


async def test_training_jobs_and_outbox_are_bound_to_the_training_queue(factory):
    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Training", slug=f"org-{uuid4()}"))
        training = enqueue(db, org_id, "agent.training")
        regular = enqueue(db, org_id, "dataset.import")
        training_id, regular_id = training.id, regular.id
    async with unit_of_work(factory, organization_id=org_id) as db:
        events = {item.job_id: item.queue for item in (await db.scalars(select(OutboxEvent))).all()}
        assert events == {training_id: "training", regular_id: "normal"}
    with pytest.raises(IntegrityError):
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(OutboxEvent(organization_id=org_id, job_id=training_id, queue="normal"))
    assert await claim(factory, training_id, org_id, frozenset({"agent.training"})) is None
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, training_id)
        assert job is not None and job.status == "queued" and job.attempts == 0
    current = await claim(factory, training_id, org_id, frozenset({"agent.training"}), "training")
    assert current is not None
    assert await fail(factory, current, error_code="test", retryable=True)
    async with unit_of_work(factory, organization_id=org_id) as db:
        events = (
            await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == training_id))
        ).all()
        assert len(events) == 2 and all(event.queue == "training" for event in events)


async def test_retry_fences_old_attempt_and_cancel_prevents_apply(factory):
    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Before", slug=f"org-{uuid4()}"))
        job_id = enqueue(db, org_id, "test").id
    first = await claim(factory, job_id, org_id, frozenset({"test"}))
    assert first is not None and await heartbeat(factory, first) is False

    async def broken_apply(db, job):
        org = await db.get(Organization, job.organization_id)
        assert org is not None
        org.name = "Should roll back"
        await db.flush()
        raise ValueError("simulated failure")

    with pytest.raises(ValueError, match="simulated"):
        await complete(factory, first, broken_apply)
    async with unit_of_work(factory, organization_id=org_id) as db:
        org, job = await db.get(Organization, org_id), await db.get(Job, job_id)
        assert org is not None and org.name == "Before"
        assert job is not None and job.status == "running"
    assert await fail(factory, first, error_code="temporary", retryable=True)
    assert await claim(factory, job_id, org_id, frozenset({"test"})) is None
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None and job.next_retry_at is not None
        job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
    second = await claim(factory, job_id, org_id, frozenset({"test"}))
    assert second is not None and second.token != first.token
    assert await heartbeat(factory, first) is None
    assert not await complete(factory, first, broken_apply)
    assert await cancel(factory, job_id, org_id)
    assert await heartbeat(factory, second) is True
    assert await complete(factory, second, broken_apply)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None and job.status == "cancelled" and job.attempts == 2
        org = await db.get(Organization, org_id)
        assert org is not None and org.name == "Before"


async def test_retry_is_bounded_and_uses_outbox(factory, monkeypatch):
    factory, tenants = factory
    monkeypatch.setattr("dialogentis.jobs.lifecycle.random.uniform", lambda low, high: 0)
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "test").id
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current = await claim(factory, job_id, org_id, frozenset({"test"}))
        assert current is not None
        assert await fail(factory, current, error_code="temporary", retryable=True)
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None and job.attempts == attempt
            if attempt < MAX_ATTEMPTS:
                assert job.status == "queued" and job.next_retry_at is not None
                assert job.next_retry_at > datetime.now(UTC)
                job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
            else:
                assert job.status == "failed" and job.error_code == "temporary"
            events = (
                await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
            ).all()
            assert len(events) == min(attempt + 1, MAX_ATTEMPTS)
    assert await claim(factory, job_id, org_id, frozenset({"test"})) is None


async def test_recovery_fences_worker_and_does_not_retry_training(factory):
    factory, tenants = factory
    org_id, other_org = uuid4(), uuid4()
    tenants.extend((org_id, other_org))
    async with unit_of_work(factory, organization_id=org_id) as db:
        retriable_id = enqueue(db, org_id, "test").id
        training_id = enqueue(db, org_id, "training").id
    async with unit_of_work(factory, organization_id=other_org) as db:
        other_id = enqueue(db, other_org, "test").id
    first = await claim(factory, retriable_id, org_id, frozenset({"test"}))
    training = await claim(factory, training_id, org_id, frozenset({"training"}))
    other = await claim(factory, other_id, other_org, frozenset({"test"}))
    assert first and training and other
    for tenant, ids in [(org_id, [retriable_id, training_id]), (other_org, [other_id])]:
        async with unit_of_work(factory, organization_id=tenant) as db:
            for job_id in ids:
                job = await db.get(Job, job_id)
                assert job is not None
                job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=3)
    counts = await asyncio.gather(
        recover_tenant(factory, org_id, frozenset({"test"})),
        recover_tenant(factory, org_id, frozenset({"test"})),
    )
    assert sum(counts) == 2
    assert await heartbeat(factory, first) is None
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, retriable_id)
        train_job = await db.get(Job, training_id)
        assert job is not None and job.status == "queued" and job.error_code == "worker_lost"
        assert (
            train_job is not None
            and train_job.status == "failed"
            and train_job.error_code == "worker_lost"
        )
        assert (
            len(
                (
                    await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == retriable_id))
                ).all()
            )
            == 2
        )
    async with unit_of_work(factory, organization_id=other_org) as db:
        other_job = await db.get(Job, other_id)
        assert other_job is not None and other_job.status == "running"


async def test_queue_age_never_fails_a_delivered_job(factory):
    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = enqueue(db, org_id, "agent.training")
        job.delivery_recoveries = MAX_DELIVERY_RECOVERIES
        job_id = job.id
        event = await db.scalar(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
        assert event is not None
        event.published_at = datetime.now(UTC) - timedelta(hours=1)
    assert await recover_tenant(factory, org_id, frozenset()) == 0
    assert await claim(factory, job_id, org_id, frozenset({"agent.training"}), "training")


async def test_rejected_deliveries_are_requeued_once_per_event_and_bounded(factory):
    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "test").id
    for number in range(MAX_DELIVERY_RECOVERIES + 1):
        async with unit_of_work(factory, organization_id=org_id) as db:
            event_id = await db.scalar(
                select(OutboxEvent.id)
                .where(OutboxEvent.job_id == job_id)
                .order_by(OutboxEvent.available_at.desc(), OutboxEvent.id.desc())
                .limit(1)
            )
            assert event_id is not None
        assert await recover_dead_letter(factory, org_id, job_id, event_id, "normal")
        assert not await recover_dead_letter(factory, org_id, job_id, event_id, "normal")
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None
            if number < MAX_DELIVERY_RECOVERIES:
                assert job.status == "queued" and job.delivery_recoveries == number + 1
            else:
                assert job.status == "failed" and job.error_code == "delivery_lost"


async def test_broker_rejection_recovers_the_linked_outbox_event(factory):
    from dialogentis.jobs.dead_letters import recover_dead_letters
    from dialogentis.jobs.tasks import broker, run_job

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "test").id
        event = await db.scalar(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
        assert event is not None
        event_id = event.id
        event.published_at = datetime.now(UTC)

    await broker.startup()
    try:
        assert broker.write_channel is not None
        ready = await broker.write_channel.get_queue("dialogentis.jobs", ensure=True)
        await run_job.kicker().with_task_id(str(event_id)).kiq(job_id, org_id)
        delivery = await ready.get(timeout=5)
        assert delivery is not None
        await delivery.reject(requeue=False)
        for _ in range(20):
            await recover_dead_letters(factory, broker)
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                if job and job.delivery_recoveries == 1:
                    assert job.next_retry_at is not None and job.status == "queued"
                    break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("RabbitMQ rejection did not schedule a new outbox event")
    finally:
        await broker.shutdown()


async def test_periodic_reconciliation_is_deduplicated_and_recovers_overdue_work(factory):
    from pathlib import Path

    from dialogentis.jobs.maintenance import reconcile_once

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "test").id
    current = await claim(factory, job_id, org_id, frozenset({"test"}))
    assert current is not None
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None
        job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=3)
    dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    window = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        results = await asyncio.gather(
            reconcile_once(factory, dispatcher, window, frozenset({"test"})),
            reconcile_once(factory, dispatcher, window, frozenset({"test"})),
        )
        assert sorted(results) == [False, True]
        assert not await reconcile_once(factory, dispatcher, window, frozenset({"test"}))
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None and job.status == "queued" and job.error_code == "worker_lost"
            assert (
                len(
                    (
                        await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
                    ).all()
                )
                == 2
            )
    finally:
        await dispatcher.kw["bind"].dispose()


async def test_worker_runs_fenced_handler_and_retries_temporary_failure(factory):
    from dialogentis.jobs.tasks import Handler, RetryableJobError, process_job

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Before", slug=f"org-{uuid4()}"))
        completed_id = enqueue(db, org_id, "ok").id
        retry_id = enqueue(db, org_id, "retry").id
        failed_id = enqueue(db, org_id, "fail").id

    async def success(_factory, current):
        async def apply(db, job):
            org = await db.get(Organization, job.organization_id)
            assert org is not None and job.execution_token == current.token
            org.name = "After"

        return apply

    async def transient(_factory, _current):
        raise RetryableJobError("private diagnostic")

    async def permanent(_factory, _current):
        raise ValueError("private diagnostic")

    handlers = {
        "ok": Handler(success, timeout_seconds=2, retryable=True),
        "retry": Handler(transient, timeout_seconds=2, retryable=True),
        "fail": Handler(permanent, timeout_seconds=2, retryable=True),
    }
    await process_job(factory, completed_id, org_id, handlers)
    await process_job(factory, retry_id, org_id, handlers)
    await process_job(factory, failed_id, org_id, handlers)
    async with unit_of_work(factory, organization_id=org_id) as db:
        done = await db.get(Job, completed_id)
        retry = await db.get(Job, retry_id)
        failed = await db.get(Job, failed_id)
        org = await db.get(Organization, org_id)
        assert org is not None and org.name == "After"
        assert done is not None and done.status == "succeeded" and done.attempts == 1
        assert (
            retry is not None and retry.status == "queued" and retry.error_code == "temporary_error"
        )
        assert retry.next_retry_at is not None
        assert (
            failed is not None
            and failed.status == "failed"
            and failed.error_code == "handler_error"
        )
        assert (
            len((await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == retry_id))).all())
            == 2
        )


async def test_worker_heartbeats_and_cooperatively_cancels_handler(factory, monkeypatch):
    from dialogentis.jobs.tasks import Handler, process_job

    factory, tenants = factory
    monkeypatch.setattr("dialogentis.jobs.tasks.HEARTBEAT_INTERVAL", 0.05)
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Test", slug=f"org-{uuid4()}"))
        job_id = enqueue(db, org_id, "slow").id
    started, stopped = asyncio.Event(), asyncio.Event()

    async def prepare(_factory, _current):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("Cancelled handler must not finish")

    task = asyncio.create_task(
        process_job(factory, job_id, org_id, {"slow": Handler(prepare, 3, True)})
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None and job.status == "running"
        original_heartbeat = job.heartbeat_at
    for _ in range(50):
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            if (
                job is not None
                and job.heartbeat_at
                and original_heartbeat
                and job.heartbeat_at > original_heartbeat
            ):
                break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("Worker did not update heartbeat")
    assert await cancel(factory, job_id, org_id)
    await asyncio.wait_for(task, timeout=2)
    assert stopped.is_set()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None and job.status == "cancelled" and job.execution_token is None


async def test_worker_times_out_before_scheduling_retry(factory):
    from dialogentis.jobs.tasks import Handler, process_job

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Test", slug=f"org-{uuid4()}"))
        job_id = enqueue(db, org_id, "slow").id
    stopped = asyncio.Event()

    async def prepare(_factory, _current):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("Timed out handler must not finish")

    await process_job(factory, job_id, org_id, {"slow": Handler(prepare, 0.1, True)})
    assert stopped.is_set()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None and job.status == "queued" and job.error_code == "timeout"
        assert job.execution_token is None and job.next_retry_at is not None


async def test_deleting_organization_fences_inflight_job_result(factory):
    from dialogentis.jobs.tasks import Handler, process_job

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Before", slug=f"org-{uuid4()}"))
        job_id = enqueue(db, org_id, "slow").id
    started, release = asyncio.Event(), asyncio.Event()

    async def prepare(_factory, _current):
        started.set()
        await release.wait()

        async def apply(db, job):
            org = await db.get(Organization, job.organization_id)
            assert org is not None
            org.name = "Should not apply"

        return apply

    task = asyncio.create_task(
        process_job(factory, job_id, org_id, {"slow": Handler(prepare, 3, True)})
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    async with unit_of_work(factory, organization_id=org_id) as db:
        org = await db.scalar(
            select(Organization).where(Organization.id == org_id).with_for_update()
        )
        assert org is not None
        org.deletion_requested_at = datetime.now(UTC)
    release.set()
    await asyncio.wait_for(task, timeout=2)
    async with unit_of_work(factory, organization_id=org_id) as db:
        org, job = await db.get(Organization, org_id), await db.get(Job, job_id)
        assert org is not None and org.name == "Before"
        assert (
            job is not None
            and job.status == "failed"
            and job.error_code == "organization_unavailable"
        )


async def test_scheduled_maintenance_task_runs_in_worker(factory, tmp_path):
    import sys

    from taskiq.schedule_sources import LabelScheduleSource

    from dialogentis.jobs.maintenance import broker as maintenance_broker

    factory, tenants = factory
    org_id = uuid4()
    tenants.append(org_id)
    async with unit_of_work(factory, organization_id=org_id) as db:
        db.add(Organization(id=org_id, name="Test", slug=f"org-{uuid4()}"))
        await db.flush()
        job_id = enqueue(db, org_id, "training").id
    storage = ArtifactStorage(tmp_path)
    async with unit_of_work(factory, organization_id=org_id) as db:
        artifact_id = (
            await reserve_artifact(db, org_id, kind="temporary", max_bytes=10, settings=Settings())
        ).id
    async with unit_of_work(factory, organization_id=org_id) as db:
        artifact = await write_artifact(db, storage, org_id, artifact_id, [b"expired"])
        artifact.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    orphan_id = uuid4()
    storage.write(org_id, orphan_id, [b"orphan"], max_bytes=10)
    orphan_path = tmp_path / str(org_id) / str(orphan_id)
    old = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(orphan_path, (old, old))
    current = await claim(factory, job_id, org_id, frozenset({"training"}))
    assert current is not None
    async with unit_of_work(factory, organization_id=org_id) as db:
        job = await db.get(Job, job_id)
        assert job is not None
        job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=3)
    source = LabelScheduleSource(maintenance_broker)
    await source.startup()
    schedules = await source.get_schedules()
    assert any(
        item.schedule_id == "jobs-reconcile" and item.cron == "* * * * *" for item in schedules
    )
    log_path = tmp_path / "maintenance.log"
    with log_path.open("w") as log:
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "worker",
            "dialogentis.jobs.maintenance:broker",
            "--workers",
            "1",
            "--ack-type",
            "manual",
            "--max-async-tasks",
            "1",
            env={**os.environ, "ARTIFACT_ROOT": str(tmp_path)},
            stdout=log,
            stderr=log,
        )
        scheduler = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "scheduler",
            "dialogentis.jobs.maintenance:scheduler",
            "--update-interval",
            "1",
            "--loop-interval",
            "1",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(300):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, job_id)
                    artifact = await db.get(Artifact, artifact_id)
                    if (
                        job is not None
                        and job.status == "failed"
                        and artifact is None
                        and not orphan_path.exists()
                    ):
                        assert job.error_code == "worker_lost"
                        assert not (tmp_path / str(org_id) / str(artifact_id)).exists()
                        break
                assert worker.returncode is None and scheduler.returncode is None, (
                    log_path.read_text()
                )
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Scheduled recovery was not processed")
        finally:
            for process in (scheduler, worker):
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10)
