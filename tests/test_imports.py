import csv
import io
import json
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
from dialogentis.datasets.csv_import import MAX_FILE_BYTES
from dialogentis.datasets.import_models import ImportTask
from dialogentis.datasets.import_worker import prepare_import, scrub_imports
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import claim, complete
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.models import AgentDraft
from tests.test_auth import account, client_for, mail_token
from tests.test_drafts import headers

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


def csv_file(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["text", "intent", "split", "entities"])
    writer.writerows(rows)
    return output.getvalue().encode()


async def make_agent(app, client):
    await account(app, client)
    org = await client.post(
        "/api/v1/organizations", json={"name": "Team", "slug": f"org-{uuid4()}"}
    )
    org_id = UUID(org.json()["id"])
    agent = await client.post(
        f"/api/v1/organizations/{org_id}/agents", json={"name": "Bot", "slug": "bot"}
    )
    agent_id = UUID(agent.json()["id"])
    intent_id, entity_id = str(uuid4()), str(uuid4())
    async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
        draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
        assert draft is not None
        config = deepcopy(draft.configuration)
        config["intents"] = [{"id": intent_id, "code": "order", "name": "Заказ"}]
        config["entities"] = [
            {
                "id": entity_id,
                "code": "city",
                "name": "Город",
                "value_type": "string",
                "method": "learned",
            }
        ]
        draft.configuration = config
    return org_id, agent_id, intent_id, entity_id


async def publish_events(app, org_id):
    async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
        for event in (await db.scalars(select(OutboxEvent))).all():
            event.published_at = datetime.now(UTC)


async def test_csv_preview_and_atomic_apply_with_unicode(app):
    async with client_for(app) as owner:
        org_id, agent_id, _, entity_id = await make_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        text = "Доставка 👩🏽‍💻 Москва"
        start = text.index("Москва")
        entities = json.dumps([{"entity": "city", "start": start, "end": start + len("Москва")}])
        file = csv_file([[text, "order", "train", entities]])
        result = await owner.post(
            f"{base}/imports/preview",
            files={"file": ("data.csv", file, "text/csv")},
            data={"mode": "merge"},
        )
        assert result.status_code == 202, result.text
        preview_id = UUID(result.json()["preview_id"])
        await process_job(app.state.session_factory, preview_id, org_id)
        preview = await owner.get(f"{base}/imports/{preview_id}")
        assert preview.status_code == 200 and preview.json()["can_apply"]
        assert preview.json()["report"]["counts"]["examples"] == 1
        assert (await owner.get(f"{base}/draft")).json()["config"]["examples"] == []
        lease = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        auth = {**headers(lease), "If-Match": '"rev-1"'}
        applied = await owner.post(f"{base}/imports/{preview_id}/apply", headers=auth)
        assert applied.status_code == 200 and applied.headers["etag"] == '"rev-2"'
        example = applied.json()["config"]["examples"][0]
        assert example["text"] == text and example["spans"][0]["entity_id"] == entity_id
        assert example["spans"][0]["start"] == start
        assert (
            await owner.post(f"{base}/imports/{preview_id}/apply", headers=auth)
        ).status_code == 409
        await publish_events(app, org_id)


async def test_invalid_stale_and_expired_previews_never_change_draft(app, tmp_path):
    async with client_for(app) as owner, client_for(app) as viewer:
        org_id, agent_id, intent_id, _ = await make_agent(app, owner)
        viewer_email, _ = await account(app, viewer)
        other = await viewer.post(
            "/api/v1/organizations", json={"name": "Other", "slug": f"org-{uuid4()}"}
        )
        other_id = other.json()["id"]
        await owner.post(
            f"/api/v1/organizations/{org_id}/invitations",
            json={"email": viewer_email, "role": "viewer"},
        )
        invite = await mail_token(app, viewer_email, "Приглашение")
        assert (
            await viewer.post("/api/v1/invitations/accept", json={"token": invite})
        ).status_code == 200
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        lease = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        auth = {**headers(lease), "If-Match": '"rev-1"'}
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            config["examples"] = [
                {"id": str(uuid4()), "text": "Заказ 42", "intent_id": intent_id, "split": "train"}
            ]
            draft.configuration = config
        duplicate = csv_file([["  ЗАКАЗ  42 ", "order", "test", "[]"]])
        assert (
            await viewer.post(f"{base}/imports/preview", files={"file": ("data.csv", duplicate)})
        ).status_code == 403
        request = await owner.post(
            f"{base}/imports/preview", files={"file": ("data.csv", duplicate)}
        )
        invalid_id = UUID(request.json()["preview_id"])
        await process_job(app.state.session_factory, invalid_id, org_id)
        report = (await owner.get(f"{base}/imports/{invalid_id}")).json()
        assert (
            not report["can_apply"]
            and report["report"]["errors"][0]["code"] == "duplicate_across_splits"
        )
        assert (
            await owner.post(f"{base}/imports/{invalid_id}/apply", headers=auth)
        ).status_code == 409
        assert (await viewer.get(f"{base}/imports/{invalid_id}")).status_code == 403
        assert (
            await viewer.get(
                f"/api/v1/organizations/{other_id}/agents/{agent_id}/imports/{invalid_id}"
            )
        ).status_code == 404
        valid = csv_file([["Другая фраза", "order", "test", "[]"]])
        created = await owner.post(f"{base}/imports/preview", files={"file": ("data.csv", valid)})
        stale_id = UUID(created.json()["preview_id"])
        await process_job(app.state.session_factory, stale_id, org_id)
        assert (await owner.get(f"{base}/imports/{stale_id}")).json()["can_apply"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            draft.revision = 2
        assert (
            await owner.post(f"{base}/imports/{stale_id}/apply", headers=auth)
        ).status_code == 409
        assert len((await owner.get(f"{base}/draft")).json()["config"]["examples"]) == 1
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            task = await db.get(ImportTask, stale_id)
            assert task is not None and task.input_artifact_id is not None
            artifact = await db.get(Artifact, task.input_artifact_id)
            assert artifact is not None
            task.expires_at = artifact.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert (await owner.get(f"{base}/imports/{stale_id}")).status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await scrub_imports(db, org_id)
            assert await cleanup_expired(db, ArtifactStorage(tmp_path), org_id) >= 1
            task = await db.get(ImportTask, stale_id)
            assert task is not None and task.input_artifact_id is None and task.candidate is None
        assert (
            await owner.post(f"{base}/imports/preview", files={"file": ("data.txt", b"text")})
        ).status_code == 422
        assert (
            await owner.post(
                f"{base}/imports/preview", files={"file": ("data.csv", b"a" * (MAX_FILE_BYTES + 1))}
            )
        ).status_code == 413
        await publish_events(app, org_id)


async def test_import_worker_stale_fencing(app):
    async with client_for(app) as owner:
        org_id, agent_id, _, _ = await make_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        file = csv_file([["Новая фраза", "order", "train", "[]"]])
        created = await owner.post(f"{base}/imports/preview", files={"file": ("data.csv", file)})
        job_id = UUID(created.json()["job_id"])
        old = await claim(app.state.session_factory, job_id, org_id, frozenset({"dataset.import"}))
        assert old is not None
        apply = await prepare_import(app.state.session_factory, old)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            job = await db.get(Job, job_id)
            assert job is not None
            job.status, job.execution_token = "queued", None
        assert not await complete(app.state.session_factory, old, apply)
        await process_job(app.state.session_factory, job_id, org_id)
        assert (await owner.get(f"{base}/imports/{job_id}")).json()["can_apply"]
        cancelled = await owner.post(f"{base}/imports/preview", files={"file": ("data.csv", file)})
        cancelled_id = UUID(cancelled.json()["job_id"])
        job_path = f"/api/v1/organizations/{org_id}/jobs/{cancelled_id}"
        assert (await owner.get(job_path)).json()["preview_id"] == str(cancelled_id)
        assert (await owner.post(f"{job_path}/cancel")).json()["status"] == "cancelled"
        await process_job(app.state.session_factory, cancelled_id, org_id)
        assert (await owner.get(f"{base}/imports/{cancelled_id}")).json()["status"] == "cancelled"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            task = await db.get(ImportTask, cancelled_id)
            assert task is not None and task.input_artifact_id is not None
            artifact = await db.get(Artifact, task.input_artifact_id)
            assert artifact is not None and artifact.expires_at is not None
            assert artifact.expires_at <= datetime.now(UTC)
            assert (
                await cleanup_expired(db, ArtifactStorage(app.state.settings.artifact_root), org_id)
                >= 1
            )
            assert task.input_artifact_id is None
        await publish_events(app, org_id)


async def test_json_export_import_replaces_only_after_preview_and_matching_revision(app):
    async with client_for(app) as owner:
        org_id, source_id, intent_id, _ = await make_agent(app, owner)
        org = f"/api/v1/organizations/{org_id}"
        target = await owner.post(f"{org}/agents", json={"name": "Target", "slug": "target"})
        target_id = UUID(target.json()["id"])
        export = await owner.post(f"{org}/agents/{source_id}/exports", json={"format": "json"})
        export_id = UUID(export.json()["job_id"])
        await process_job(app.state.session_factory, export_id, org_id)
        document = await owner.get(f"{org}/exports/{export_id}/download")
        assert document.status_code == 200
        base = f"{org}/agents/{target_id}"
        preview = await owner.post(
            f"{base}/imports/preview",
            files={"file": ("agent.json", document.content)},
            data={"mode": "replace"},
        )
        assert preview.status_code == 202
        preview_id = UUID(preview.json()["preview_id"])
        await process_job(app.state.session_factory, preview_id, org_id)
        report = (await owner.get(f"{base}/imports/{preview_id}")).json()
        assert report["can_apply"] and report["report"]["counts"]["intents"] == 1
        lease = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        bad = {**headers(lease), "If-Match": '"rev-9"'}
        assert (
            await owner.post(f"{base}/imports/{preview_id}/apply", headers=bad)
        ).status_code == 412
        auth = {**headers(lease), "If-Match": '"rev-1"'}
        result = await owner.post(f"{base}/imports/{preview_id}/apply", headers=auth)
        assert result.status_code == 200
        assert result.json()["config"]["intents"][0]["id"] == intent_id
        assert result.json()["config"]["flow"] == document.json()["config"]["flow"]
        await publish_events(app, org_id)
