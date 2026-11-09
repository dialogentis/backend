import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.api import create_app
from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import cleanup_expired
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings, get_settings
from dialogentis.datasets.export_models import ExportTask
from dialogentis.datasets.export_worker import prepare_export, scrub_exports
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import claim, complete
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.models import AgentDraft
from tests.test_auth import account, client_for, mail_token

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


@pytest.fixture
async def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    application = create_app(Settings(public_base_url="http://test", artifact_root=tmp_path))
    async with application.router.lifespan_context(application):
        yield application
    get_settings.cache_clear()


async def mark_published(app, org_id):
    async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
        for event in (await db.scalars(select(OutboxEvent))).all():
            event.published_at = datetime.now(UTC)


async def test_export_snapshot_permissions_expiry_and_cleanup(app, tmp_path):
    async with client_for(app) as owner, client_for(app) as editor:
        await account(app, owner)
        editor_email, editor_id = await account(app, editor)
        org = await owner.post(
            "/api/v1/organizations", json={"name": "Team", "slug": f"org-{uuid4()}"}
        )
        org_id = UUID(org.json()["id"])
        other = await editor.post(
            "/api/v1/organizations", json={"name": "Other", "slug": f"org-{uuid4()}"}
        )
        other_id = other.json()["id"]
        await owner.post(
            f"/api/v1/organizations/{org_id}/invitations",
            json={"email": editor_email, "role": "editor"},
        )
        token = await mail_token(app, editor_email, "Приглашение")
        assert (
            await editor.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 200
        created = await owner.post(
            f"/api/v1/organizations/{org_id}/agents", json={"name": "Bot", "slug": "bot"}
        )
        agent_id = UUID(created.json()["id"])
        base = f"/api/v1/organizations/{org_id}"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            config["flow"]["nodes"][2]["config"]["text"] = "Исходный ответ"
            draft.configuration = config
        response = await editor.post(
            f"{base}/agents/{agent_id}/exports", json={"format": "json", "source": "draft"}
        )
        assert response.status_code == 202, response.text
        job_id = UUID(response.json()["job_id"])
        assert response.json()["revision"] == 1
        assert (await editor.get(f"{base}/jobs/{job_id}")).json()["status"] == "queued"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            config["flow"]["nodes"][2]["config"]["text"] = "Изменённый ответ"
            draft.configuration, draft.revision = config, 2
        await process_job(app.state.session_factory, job_id, org_id)
        await process_job(app.state.session_factory, job_id, org_id)
        status = await editor.get(f"{base}/jobs/{job_id}")
        assert status.status_code == 200 and status.json()["status"] == "succeeded"
        downloaded = await editor.get(f"{base}/exports/{job_id}/download")
        assert downloaded.status_code == 200
        assert downloaded.json()["config"]["flow"]["nodes"][2]["config"]["text"] == "Исходный ответ"
        assert (
            downloaded.headers["content-disposition"] == 'attachment; filename="dialogentis.json"'
        )
        assert (
            await editor.get(f"/api/v1/organizations/{other_id}/exports/{job_id}/download")
        ).status_code == 404
        assert (
            await owner.patch(f"{base}/members/{editor_id}", json={"role": "viewer"})
        ).status_code == 200
        assert (await editor.get(f"{base}/exports/{job_id}/download")).status_code == 403
        assert (await editor.get(f"{base}/jobs/{job_id}")).status_code == 403
        assert (
            await editor.post(f"{base}/agents/{agent_id}/exports", json={"format": "csv"})
        ).status_code == 403
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            task = await db.get(ExportTask, job_id)
            artifacts = (await db.scalars(select(Artifact).where(Artifact.job_id == job_id))).all()
            assert task is not None and task.snapshot is None and len(artifacts) == 1
            task.download_until = datetime.now(UTC) - timedelta(seconds=1)
            artifacts[0].expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert (await owner.get(f"{base}/exports/{job_id}/download")).status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await cleanup_expired(db, ArtifactStorage(tmp_path), org_id) == 1
            task = await db.get(ExportTask, job_id)
            assert task is not None and task.artifact_id is None
        await mark_published(app, org_id)


async def test_export_cancellation_fencing_and_failed_snapshot_cleanup(app):
    async with client_for(app) as owner:
        await account(app, owner)
        org = await owner.post(
            "/api/v1/organizations", json={"name": "Team", "slug": f"org-{uuid4()}"}
        )
        org_id = UUID(org.json()["id"])
        created = await owner.post(
            f"/api/v1/organizations/{org_id}/agents", json={"name": "Bot", "slug": "bot"}
        )
        agent_id = UUID(created.json()["id"])
        base = f"/api/v1/organizations/{org_id}"
        endpoint = f"{base}/agents/{agent_id}/exports"
        cancelled = await owner.post(endpoint, json={"format": "csv"})
        cancel_id = UUID(cancelled.json()["job_id"])
        assert (await owner.post(f"{base}/jobs/{cancel_id}/cancel")).json()["status"] == "cancelled"
        await process_job(app.state.session_factory, cancel_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            task = await db.get(ExportTask, cancel_id)
            assert task is not None and task.snapshot is None
        running = await owner.post(endpoint, json={"format": "csv"})
        running_id = UUID(running.json()["job_id"])
        current = await claim(
            app.state.session_factory, running_id, org_id, frozenset({"dataset.export"})
        )
        assert current is not None
        running_apply = await prepare_export(app.state.session_factory, current)
        cancelled = await owner.post(f"{base}/jobs/{running_id}/cancel")
        assert cancelled.json()["cancel_requested"] is True
        assert await complete(app.state.session_factory, current, running_apply)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            job = await db.get(Job, running_id)
            assert job is not None and job.status == "cancelled"
            assert (
                await db.scalars(select(Artifact).where(Artifact.job_id == running_id))
            ).all() == []
            await scrub_exports(db, org_id)
            task = await db.get(ExportTask, running_id)
            assert task is not None and task.snapshot is None
        queued = await owner.post(endpoint, json={"format": "csv"})
        job_id = UUID(queued.json()["job_id"])
        old = await claim(app.state.session_factory, job_id, org_id, frozenset({"dataset.export"}))
        assert old is not None
        apply = await prepare_export(app.state.session_factory, old)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None
            job.status, job.execution_token = "queued", None
        assert not await complete(app.state.session_factory, old, apply)
        await process_job(app.state.session_factory, job_id, org_id)
        result = await owner.get(f"{base}/exports/{job_id}/download")
        assert result.status_code == 200 and result.text.startswith("text,intent,split,entities")
        invalid = await owner.post(endpoint, json={"format": "json"})
        invalid_id = UUID(invalid.json()["job_id"])
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            task = await db.get(ExportTask, invalid_id)
            assert task is not None and task.snapshot is not None
            task.snapshot = {**task.snapshot, "slots": [{"secret": "not-exported"}]}
        await process_job(app.state.session_factory, invalid_id, org_id)
        status = (await owner.get(f"{base}/jobs/{invalid_id}")).json()
        assert (
            status["status"] == "failed" and status["error_code"] == "invalid_export_configuration"
        )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await scrub_exports(db, org_id)
            task = await db.get(ExportTask, invalid_id)
            assert task is not None and task.snapshot is None
        await mark_published(app, org_id)
