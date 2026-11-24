import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import read_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import cancel, recover_tenant
from dialogentis.jobs.models import Job
from dialogentis.jobs.tasks import Handler, process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.nlu.model_store import ModelIdentity, verify_dialogue_manifest
from dialogentis.training.serial import run_serialized
from dialogentis.training.worker import prepare_training
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import version_status
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def training_job(app, client):
    user_id, org_id, agent_id = await create_agent(app, client)
    await populate(app, org_id, agent_id)
    result = await prepare(app, user_id, org_id, agent_id, 1, f"key-{uuid4()}")
    return org_id, agent_id, result.version.id, result.version.job_id


async def test_preparation_produces_fenced_signed_snapshot(app):
    async with client_for(app) as client:
        org_id, agent_id, version_id, job_id = await training_job(app, client)
        factory = app.state.session_factory
        await process_job(factory, job_id, org_id, HANDLERS, "training")
        await process_job(factory, job_id, org_id, HANDLERS, "training")
        async with unit_of_work(factory, organization_id=org_id) as db:
            version = await db.get(AgentVersion, version_id)
            job = await db.get(Job, job_id)
            assert version is not None and job is not None
            assert job.status == "succeeded" and job.attempts == 1 and job.stage == "packaging"
            assert version_status(version, job) == "ready"
            assert version.validation_report is not None
            assert version.validation_report["graph_validation"] == "passed"
            assert version.model_artifact_id is not None
            artifacts = (await db.scalars(select(Artifact).where(Artifact.job_id == job_id))).all()
            assert len(artifacts) == 1 and artifacts[0].id == version.model_artifact_id
            identity = ModelIdentity(org_id, agent_id, version_id, version.model_artifact_id)
            manifest = await read_artifact(
                db,
                ArtifactStorage(app.state.settings.artifact_root),
                org_id,
                identity.artifact_id,
                max_bytes=1024,
                expected_kind="model",
            )
            verify_dialogue_manifest(manifest, identity, version.config_hash, app.state.settings)


async def test_training_completes_through_rabbitmq_and_separate_worker(app, tmp_path):
    async with client_for(app) as client:
        org_id, _, version_id, job_id = await training_job(app, client)
        factory = app.state.session_factory
        log_path = tmp_path / "training-worker.log"
        with log_path.open("w") as log:
            worker = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "taskiq",
                "worker",
                "dialogentis.jobs.training_tasks:broker",
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
                async with asyncio.timeout(40):
                    while True:
                        async with unit_of_work(factory, organization_id=org_id) as db:
                            job = await db.get(Job, job_id)
                            version = await db.get(AgentVersion, version_id)
                            assert job is not None and version is not None
                            if job.status == "succeeded":
                                assert version_status(version, job) == "ready"
                                assert version.validation_report is not None
                                assert version.validation_report["graph_validation"] == "passed"
                                break
                            assert job.status != "failed", job.error_code
                        assert worker.returncode is None and dispatcher.returncode is None, (
                            log_path.read_text()
                        )
                        await asyncio.sleep(0.1)
            finally:
                for process in (dispatcher, worker):
                    if process.returncode is None:
                        process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)


async def test_cancel_during_preparation_and_never_publishes(app, monkeypatch):
    monkeypatch.setattr("dialogentis.jobs.tasks.HEARTBEAT_INTERVAL", 0.05)
    started = asyncio.Event()

    async def slow_stage(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("dialogentis.training.worker.set_stage", slow_stage)
    async with client_for(app) as client:
        org_id, _, version_id, job_id = await training_job(app, client)
        factory = app.state.session_factory
        task = asyncio.create_task(process_job(factory, job_id, org_id, HANDLERS, "training"))
        async with asyncio.timeout(10):
            await started.wait()
            await asyncio.sleep(0.2)
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                assert (
                    job is not None and job.heartbeat_at is not None and job.started_at is not None
                )
                assert job.heartbeat_at > job.started_at
            assert await cancel(factory, job_id, org_id)
            await task
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            version = await db.get(AgentVersion, version_id)
            assert job is not None and job.status == "cancelled"
            assert version is not None and version.model_artifact_id is None
            assert (await db.scalars(select(Artifact).where(Artifact.job_id == job_id))).all() == []


async def test_lost_training_worker_is_fenced_without_retry(app, monkeypatch):
    monkeypatch.setattr("dialogentis.jobs.tasks.HEARTBEAT_INTERVAL", 0.5)
    started = asyncio.Event()

    async def slow_stage(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("dialogentis.training.worker.set_stage", slow_stage)
    async with client_for(app) as client:
        org_id, _, version_id, job_id = await training_job(app, client)
        factory = app.state.session_factory
        task = asyncio.create_task(process_job(factory, job_id, org_id, HANDLERS, "training"))
        async with asyncio.timeout(10):
            await started.wait()
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                assert job is not None
                job.heartbeat_at = datetime.now(UTC) - timedelta(seconds=91)
            assert await recover_tenant(factory, org_id, frozenset()) == 1
            await task
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            version = await db.get(AgentVersion, version_id)
            assert job is not None and job.status == "failed" and job.error_code == "worker_lost"
            assert job.attempts == 1 and version is not None and version.model_artifact_id is None


async def test_training_timeout_fails_without_retry(app, monkeypatch):

    async def slow_stage(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr("dialogentis.training.worker.set_stage", slow_stage)
    async with client_for(app) as client:
        org_id, _, version_id, job_id = await training_job(app, client)
        handlers = {
            "agent.training": Handler(prepare_training, timeout_seconds=1.5, retryable=False)
        }
        await process_job(app.state.session_factory, job_id, org_id, handlers, "training")
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            version = await db.get(AgentVersion, version_id)
            assert job is not None and job.status == "failed" and job.error_code == "timeout"
            assert job.attempts == 1 and version is not None and version.model_artifact_id is None


async def test_full_model_quota_does_not_publish_partial_result(app, monkeypatch):
    monkeypatch.setenv("ARTIFACT_QUOTA_BYTES", "1")
    get_settings.cache_clear()
    async with client_for(app) as client:
        org_id, _, version_id, job_id = await training_job(app, client)
        factory = app.state.session_factory
        await process_job(factory, job_id, org_id, HANDLERS, "training")
        async with unit_of_work(factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            version = await db.get(AgentVersion, version_id)
            assert (
                job is not None
                and job.status == "failed"
                and job.error_code == "artifact_quota_exceeded"
            )
            assert version is not None and version.model_artifact_id is None
            assert (await db.scalars(select(Artifact).where(Artifact.job_id == job_id))).all() == []


async def test_platform_training_slot_serializes_and_releases_after_cancel(app):
    factory = app.state.session_factory
    first_started, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def first():
        first_started.set()
        await release.wait()

    async def second():
        second_started.set()

    first_task = asyncio.create_task(run_serialized(factory, first))
    await first_started.wait()
    second_task = asyncio.create_task(run_serialized(factory, second))
    await asyncio.sleep(0.5)
    assert not second_started.is_set()
    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task
    release.set()
    await first_task
    async with asyncio.timeout(5):
        await run_serialized(factory, second)
    assert second_started.is_set()
