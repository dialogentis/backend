import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aio_pika import DeliveryMode, Message
from sqlalchemy import delete, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from taskiq import Context

from dialogentis.config import Settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.dispatcher import dispatch_once, dispatcher_factory
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job, OutboxEvent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_enqueue_is_atomic_and_tenant_scoped():
    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    first, second = uuid4(), uuid4()
    try:
        with pytest.raises(RuntimeError, match="abort"):
            async with unit_of_work(factory, organization_id=first) as db:
                enqueue(db, first, "test")
                await db.flush()
                raise RuntimeError("abort")
        async with unit_of_work(factory, organization_id=first) as db:
            assert (await db.scalars(select(Job))).all() == []
            assert (await db.scalars(select(OutboxEvent))).all() == []
            first_job = enqueue(db, first, "test")
            first_id = first_job.id
        async with unit_of_work(factory, organization_id=second) as db:
            second_job = enqueue(db, second, "test")
            second_id = second_job.id
        async with unit_of_work(factory) as db:
            assert (await db.scalars(select(Job))).all() == []
            assert (await db.scalars(select(OutboxEvent))).all() == []
        async with unit_of_work(factory, organization_id=first) as db:
            assert (await db.scalars(select(Job.id))).all() == [first_id]
            assert (await db.scalars(select(OutboxEvent.job_id))).all() == [first_id]
            assert await db.get(Job, second_id) is None
        with pytest.raises(IntegrityError):
            async with unit_of_work(factory, organization_id=first) as db:
                db.add(OutboxEvent(organization_id=first, job_id=second_id))
        with pytest.raises(DBAPIError):
            async with unit_of_work(factory, organization_id=first) as db:
                enqueue(db, second, "test")
        async with unit_of_work(factory, organization_id=second) as db:
            assert (await db.scalars(select(Job.id))).all() == [second_id]
            assert (await db.scalars(select(OutboxEvent.job_id))).all() == [second_id]
    finally:
        for organization_id in (first, second):
            async with unit_of_work(factory, organization_id=organization_id) as db:
                await db.execute(delete(OutboxEvent))
                await db.execute(delete(Job))
        await factory.kw["bind"].dispose()


async def test_dispatcher_is_restricted_and_recovers_after_publish_failure(tmp_path: Path):
    from tests.broker_tasks import broker, job_probe

    url = Settings().database_url
    assert url is not None
    app_factory = session_factory(url.get_secret_value())
    dispatch_factory = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    organization_id = uuid4()
    async with unit_of_work(app_factory, organization_id=organization_id) as db:
        ready_id = enqueue(db, organization_id, "test").id
        future_id = enqueue(db, organization_id, "test").id
        await db.flush()
        event = await db.scalar(select(OutboxEvent).where(OutboxEvent.job_id == future_id))
        assert event is not None
        event.available_at = datetime.now(UTC) + timedelta(hours=1)
    try:
        async with dispatch_factory() as db:
            assert await db.scalar(text("SELECT current_user")) == "dialogentis_dispatcher"
            assert (
                len(
                    (
                        await db.scalars(
                            select(OutboxEvent).where(
                                OutboxEvent.organization_id == organization_id
                            )
                        )
                    ).all()
                )
                == 2
            )
        with pytest.raises(DBAPIError):
            async with dispatch_factory() as db:
                await db.execute(select(Job))
        with pytest.raises(DBAPIError):
            async with dispatch_factory() as db:
                await db.execute(text("SELECT * FROM organizations"))
        with pytest.raises(DBAPIError):
            async with dispatch_factory() as db:
                await db.execute(text("UPDATE job_outbox SET organization_id = gen_random_uuid()"))
        with pytest.raises(DBAPIError):
            async with dispatch_factory() as db:
                await db.execute(text("UPDATE job_outbox SET queue = 'training'"))

        uncertain = []

        async def fail_publish(job_id, org_id, queue):
            assert queue == "normal"
            uncertain.append((job_id, org_id))
            raise ConnectionError("publication confirmation lost")

        with pytest.raises(ConnectionError):
            await dispatch_once(dispatch_factory, fail_publish)
        async with unit_of_work(app_factory, organization_id=organization_id) as db:
            rows = (await db.scalars(select(OutboxEvent))).all()
            assert all(row.published_at is None for row in rows)

        log_path = tmp_path / "worker.log"
        with log_path.open("w") as log:
            worker = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "taskiq",
                "worker",
                "tests.broker_tasks:broker",
                "--workers",
                "1",
                "--ack-type",
                "manual",
                "--max-async-tasks",
                "1",
                env={**os.environ, "TEST_OUTPUT_DIR": str(tmp_path)},
                stdout=log,
                stderr=log,
            )
            try:
                await broker.startup()
                assert broker.write_channel is not None and broker.write_channel.publisher_confirms

                async def publish(job_id, org_id, queue):
                    assert queue == "normal"
                    assert (job_id, org_id) == uncertain[0]
                    await job_probe.kiq(job_id, org_id)

                assert await dispatch_once(dispatch_factory, publish) == 1
                assert await dispatch_once(dispatch_factory, publish) == 0
                for _ in range(100):
                    if (tmp_path / str(ready_id)).exists():
                        break
                    assert worker.returncode is None, log_path.read_text()
                    await asyncio.sleep(0.1)
                assert (tmp_path / str(ready_id)).read_text() == str(organization_id)
                assert not (tmp_path / str(future_id)).exists()
                async with unit_of_work(app_factory, organization_id=organization_id) as db:
                    rows = {
                        row.job_id: row for row in (await db.scalars(select(OutboxEvent))).all()
                    }
                    assert rows[ready_id].published_at is not None
                    assert rows[future_id].published_at is None
            finally:
                await broker.shutdown()
                worker.terminate()
                await asyncio.wait_for(worker.wait(), timeout=10)
    finally:
        await dispatch_factory.kw["bind"].dispose()
        await app_factory.kw["bind"].dispose()


async def test_worker_commits_terminal_state_once_before_ack(tmp_path: Path):
    from dialogentis.jobs.tasks import broker, run_job

    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
        cancelled = enqueue(db, org_id, "unsupported")
        cancelled.cancel_requested = True
        cancelled_id = cancelled.id
    log_path = tmp_path / "worker.log"
    with log_path.open("w") as log:
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "worker",
            "dialogentis.jobs.tasks:broker",
            "--workers",
            "1",
            "--ack-type",
            "manual",
            "--max-async-tasks",
            "1",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        try:
            await broker.startup()
            await run_job.kiq(job_id, org_id)
            await run_job.kiq(job_id, org_id)
            await run_job.kiq(cancelled_id, org_id)
            for _ in range(100):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    current = await db.get(Job, cancelled_id)
                    if current is not None and current.status == "cancelled":
                        break
                assert worker.returncode is None, log_path.read_text()
                await asyncio.sleep(0.1)
            async with unit_of_work(factory, organization_id=org_id) as db:
                first = await db.get(Job, job_id)
                last = await db.get(Job, cancelled_id)
                assert first is not None and first.status == "failed" and first.attempts == 1
                assert first.error_code == "unsupported_job_kind" and first.finished_at is not None
                assert last is not None and last.status == "cancelled" and last.attempts == 0
        finally:
            await broker.shutdown()
            worker.terminate()
            await asyncio.wait_for(worker.wait(), timeout=10)
            await factory.kw["bind"].dispose()


async def test_ack_failure_keeps_committed_state_and_closes_consumer(monkeypatch):
    from dialogentis.jobs.tasks import broker, run_job

    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
    channel = SimpleNamespace(close=AsyncMock())
    context = cast(
        Context, SimpleNamespace(ack=AsyncMock(side_effect=ConnectionError("ack failed")))
    )
    monkeypatch.setattr(broker, "read_channel", channel)
    try:
        with pytest.raises(ConnectionError, match="ack failed"):
            await run_job.original_func(job_id, org_id, context)
        channel.close.assert_awaited_once()
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None and job.status == "failed" and job.attempts == 1
        success_ack = AsyncMock()
        await run_job.original_func(job_id, org_id, cast(Context, SimpleNamespace(ack=success_ack)))
        success_ack.assert_awaited_once()
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None and job.attempts == 1
    finally:
        await factory.kw["bind"].dispose()


async def test_worker_redelivers_when_database_write_cannot_start(tmp_path: Path):
    from dialogentis.jobs.tasks import broker, run_job

    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
    bad_path, good_path = tmp_path / "bad.log", tmp_path / "good.log"
    command = (
        sys.executable,
        "-m",
        "taskiq",
        "worker",
        "dialogentis.jobs.tasks:broker",
        "--workers",
        "1",
        "--ack-type",
        "manual",
        "--max-async-tasks",
        "1",
    )
    with bad_path.open("w") as bad_log, good_path.open("w") as good_log:
        bad = await asyncio.create_subprocess_exec(
            *command,
            env={**os.environ, "DATABASE_URL": "postgresql://invalid:invalid@postgres/dialogentis"},
            stdout=bad_log,
            stderr=bad_log,
        )
        good = None
        try:
            await broker.startup()
            await run_job.kiq(job_id, org_id)
            for _ in range(150):
                if "Exception found while executing function" in bad_path.read_text():
                    break
                assert bad.returncode is None, bad_path.read_text()
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The failing worker did not receive the job")
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                assert job is not None and job.status == "queued"
            good = await asyncio.create_subprocess_exec(
                *command,
                env=os.environ.copy(),
                stdout=good_log,
                stderr=good_log,
            )
            for _ in range(150):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, job_id)
                    if job is not None and job.status == "failed":
                        assert job.attempts == 1
                        break
                assert good.returncode is None, good_path.read_text()
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The unacknowledged job was not redelivered")
        finally:
            await broker.shutdown()
            for process in (bad, good):
                if process is not None:
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
            await factory.kw["bind"].dispose()


async def test_worker_redelivers_after_commit_failure(tmp_path: Path):
    from dialogentis.jobs.tasks import broker, run_job

    settings = Settings()
    assert settings.database_url is not None and settings.migration_database_url is not None
    factory = session_factory(settings.database_url.get_secret_value())
    migrator = session_factory(settings.migration_database_url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
    async with migrator() as db, db.begin():
        await db.execute(
            text(f"""
            CREATE FUNCTION test_reject_job_commit() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.id = '{job_id}'::uuid THEN
                    RAISE EXCEPTION 'simulated commit failure';
                END IF;
                RETURN NEW;
            END $$
        """)
        )
        await db.execute(
            text("""
            CREATE CONSTRAINT TRIGGER test_reject_job_commit
            AFTER UPDATE ON jobs DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW EXECUTE FUNCTION test_reject_job_commit()
        """)
        )
    command = (
        sys.executable,
        "-m",
        "taskiq",
        "worker",
        "dialogentis.jobs.tasks:broker",
        "--workers",
        "1",
        "--ack-type",
        "manual",
        "--max-async-tasks",
        "1",
    )
    bad_path, good_path = tmp_path / "commit-failure.log", tmp_path / "recovered.log"
    with bad_path.open("w") as bad_log, good_path.open("w") as good_log:
        bad = await asyncio.create_subprocess_exec(
            *command,
            env=os.environ.copy(),
            stdout=bad_log,
            stderr=bad_log,
        )
        good = None
        try:
            await broker.startup()
            await run_job.kiq(job_id, org_id)
            for _ in range(150):
                if "Exception found while executing function" in bad_path.read_text():
                    break
                assert bad.returncode is None, bad_path.read_text()
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The commit failure was not observed")
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                assert job is not None and job.status == "queued" and job.attempts == 0
            async with migrator() as db, db.begin():
                await db.execute(text("DROP TRIGGER test_reject_job_commit ON jobs"))
            good = await asyncio.create_subprocess_exec(
                *command,
                env=os.environ.copy(),
                stdout=good_log,
                stderr=good_log,
            )
            for _ in range(150):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, job_id)
                    if job is not None and job.status == "failed":
                        assert job.attempts == 1
                        break
                assert good.returncode is None, good_path.read_text()
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The message was not redelivered after commit failure")
        finally:
            await broker.shutdown()
            for process in (bad, good):
                if process is not None:
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
            async with migrator() as db, db.begin():
                await db.execute(text("DROP TRIGGER IF EXISTS test_reject_job_commit ON jobs"))
                await db.execute(text("DROP FUNCTION IF EXISTS test_reject_job_commit()"))
            await migrator.kw["bind"].dispose()
            await factory.kw["bind"].dispose()


async def test_dispatcher_process_delivers_job_to_worker(tmp_path: Path):
    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
    log_path = tmp_path / "pipeline.log"
    with log_path.open("w") as log:
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "worker",
            "dialogentis.jobs.tasks:broker",
            "--workers",
            "1",
            "--ack-type",
            "manual",
            "--max-async-tasks",
            "1",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        dispatcher = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dialogentis.jobs.dispatcher",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(150):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, job_id)
                    event = await db.scalar(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
                    if job is not None and job.status == "failed" and event and event.published_at:
                        assert job.attempts == 1
                        break
                assert worker.returncode is None and dispatcher.returncode is None, (
                    log_path.read_text()
                )
                await asyncio.sleep(0.1)
            else:
                pytest.fail("The dispatcher did not deliver the job")
        finally:
            for process in (dispatcher, worker):
                if process.returncode is None:
                    process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10)
            await factory.kw["bind"].dispose()


async def test_training_and_normal_jobs_use_separate_real_workers(tmp_path: Path):
    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        normal_id = enqueue(db, org_id, "unsupported").id
        training_id = enqueue(db, org_id, "agent.training").id
    log_path = tmp_path / "queues.log"
    with log_path.open("w") as log:
        common = [sys.executable, "-m", "taskiq", "worker"]
        flags = ["--workers", "1", "--ack-type", "manual", "--max-async-tasks", "1"]
        normal = await asyncio.create_subprocess_exec(
            *common,
            "dialogentis.jobs.tasks:broker",
            *flags,
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        dispatcher = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dialogentis.jobs.dispatcher",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        training = None
        try:
            for _ in range(150):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    regular = await db.get(Job, normal_id)
                    pending = await db.get(Job, training_id)
                    event = await db.scalar(
                        select(OutboxEvent).where(OutboxEvent.job_id == training_id)
                    )
                    if (
                        regular
                        and regular.status == "failed"
                        and pending
                        and event
                        and event.published_at
                    ):
                        assert pending.status == "queued" and pending.attempts == 0
                        break
                assert normal.returncode is None and dispatcher.returncode is None, (
                    log_path.read_text()
                )
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Dispatcher did not publish both jobs")
            training = await asyncio.create_subprocess_exec(
                *common,
                "dialogentis.jobs.training_tasks:broker",
                *flags,
                env=os.environ.copy(),
                stdout=log,
                stderr=log,
            )
            for _ in range(150):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, training_id)
                    if job and job.status == "failed":
                        assert job.error_code == "unsupported_job_kind" and job.attempts == 1
                        break
                assert training.returncode is None, log_path.read_text()
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Training worker did not consume its queue")
        finally:
            for process in (dispatcher, normal, training):
                if process is not None:
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
            async with unit_of_work(factory, organization_id=org_id) as db:
                await db.execute(delete(OutboxEvent))
                await db.execute(delete(Job))
            await factory.kw["bind"].dispose()


async def test_invalid_messages_go_to_dead_letter_queue(tmp_path: Path):
    from dialogentis.jobs.tasks import broker, run_job

    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    org_id = uuid4()
    async with unit_of_work(factory, organization_id=org_id) as db:
        job_id = enqueue(db, org_id, "unsupported").id
    log_path = tmp_path / "poison.log"
    with log_path.open("w") as log:
        worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "worker",
            "dialogentis.jobs.tasks:broker",
            "--workers",
            "1",
            "--ack-type",
            "manual",
            "--max-async-tasks",
            "1",
            env=os.environ.copy(),
            stdout=log,
            stderr=log,
        )
        try:
            await broker.startup()
            assert broker.write_channel is not None
            invalid = [
                b"not-a-task-payload",
                json.dumps(
                    {
                        "task_id": "1",
                        "task_name": "unknown",
                        "labels": {"ack_type": "manual"},
                        "args": [],
                        "kwargs": {},
                    }
                ).encode(),
                json.dumps(
                    {
                        "task_id": "2",
                        "task_name": run_job.task_name,
                        "labels": {"ack_type": "manual"},
                        "args": ["not-a-uuid", str(org_id)],
                        "kwargs": {},
                    }
                ).encode(),
                json.dumps(
                    {
                        "task_id": "3",
                        "task_name": run_job.task_name,
                        "labels": {"ack_type": "when_received"},
                        "args": [str(job_id), str(org_id)],
                        "kwargs": {},
                    }
                ).encode(),
                json.dumps(
                    {
                        "task_id": "4",
                        "task_name": run_job.task_name,
                        "labels": {"ack_type": "manual"},
                        "args": [str(job_id), str(org_id)],
                        "kwargs": {"context": "forged"},
                    }
                ).encode(),
            ]
            for payload in invalid:
                await broker.write_channel.default_exchange.publish(
                    Message(payload, delivery_mode=DeliveryMode.PERSISTENT),
                    routing_key="dialogentis.jobs",
                )
            await run_job.kiq(job_id, org_id)
            dead = await broker.write_channel.get_queue("dialogentis.jobs.dead", ensure=True)
            received = []
            for _ in range(150):
                message = await dead.get(fail=False)
                if message is not None:
                    received.append(message.body)
                    await message.ack()
                if len(received) == len(invalid):
                    break
                assert worker.returncode is None, log_path.read_text()
                await asyncio.sleep(0.1)
            assert sorted(received) == sorted(invalid)
            for _ in range(100):
                async with unit_of_work(factory, organization_id=org_id) as db:
                    job = await db.get(Job, job_id)
                    if job is not None and job.status == "failed":
                        break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Worker stopped processing valid jobs after a malformed message")
            assert "not-a-task-payload" not in log_path.read_text()
        finally:
            await broker.shutdown()
            if worker.returncode is None:
                worker.terminate()
            await asyncio.wait_for(worker.wait(), timeout=10)
            await factory.kw["bind"].dispose()
