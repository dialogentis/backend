import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete, select

from dialogentis.api import create_app
from dialogentis.artifacts import ArtifactStorage
from dialogentis.auth.models import EmailOutbox
from dialogentis.config import Settings, get_settings
from dialogentis.db import unit_of_work
from dialogentis.deletion_journal import read_markers, write_marker
from dialogentis.jobs.lifecycle import cancel
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.models import Agent, AgentDraft, Membership, Organization, User
from dialogentis.organizations import OrganizationDeletion
from dialogentis.replay_deletions import replay
from tests.test_auth import PASSWORD, account, client_for, mail_token

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


@pytest.fixture
async def app():
    application = create_app(Settings(public_base_url="http://test"))
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture(autouse=True)
def clear_cached_settings():
    yield
    get_settings.cache_clear()


async def test_deletion_is_atomic_authorized_and_replayed_after_restore(app, tmp_path, monkeypatch):
    from dialogentis.jobs.tasks import process_job

    monkeypatch.setenv("DELETION_JOURNAL_ROOT", str(tmp_path))
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setenv("ARTIFACT_ROOT", str(artifact_root))
    get_settings.cache_clear()
    async with (
        client_for(app) as owner,
        client_for(app) as owner2,
        client_for(app) as guest,
        client_for(app) as pending,
    ):
        owner_email, owner_id = await account(app, owner)
        guest_email, guest_id = await account(app, guest)
        pending_email, _ = await account(app, pending)
        login = await owner2.post(
            "/api/v1/auth/login", json={"email": owner_email, "password": PASSWORD}
        )
        assert login.status_code == 200
        owner2.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        first = await owner.post(
            "/api/v1/organizations", json={"name": "Удаляемая", "slug": f"org-{uuid4()}"}
        )
        second = await guest.post(
            "/api/v1/organizations", json={"name": "Сохранить", "slug": f"org-{uuid4()}"}
        )
        assert first.status_code == second.status_code == 201
        org_id, other_id = UUID(first.json()["id"]), UUID(second.json()["id"])
        storage = ArtifactStorage(artifact_root)
        file_id, other_file_id = uuid4(), uuid4()
        storage.write(org_id, file_id, [b"delete me"], max_bytes=100)
        other_info = storage.write(other_id, other_file_id, [b"keep me"], max_bytes=100)
        invitation = await owner.post(
            f"/api/v1/organizations/{org_id}/invitations",
            json={"email": guest_email, "role": "viewer"},
        )
        assert invitation.status_code == 201
        token = await mail_token(app, guest_email, "Приглашение")
        assert (
            await guest.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 200
        assert (
            await owner.post(
                f"/api/v1/organizations/{org_id}/invitations",
                json={"email": pending_email, "role": "editor"},
            )
        ).status_code == 201
        pending_token = await mail_token(app, pending_email, "Приглашение")
        agent_id = uuid4()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            agent = Agent(id=agent_id, organization_id=org_id, name="Agent", slug="agent")
            db.add(agent)
            await db.flush()
            db.add(AgentDraft(organization_id=org_id, agent_id=agent_id, configuration={}))
        async with unit_of_work(app.state.session_factory, organization_id=other_id) as db:
            other_agent_id = uuid4()
            db.add(Agent(id=other_agent_id, organization_id=other_id, name="Keep", slug="agent"))
        endpoint = f"/api/v1/organizations/{org_id}/deletion"
        body = {"password": PASSWORD, "confirmation_name": "Удаляемая"}
        assert (await guest.post(endpoint, json=body)).status_code == 403
        assert (
            await owner.post(endpoint, json=body, headers={"X-CSRF-Token": "bad"})
        ).status_code == 403
        assert (
            await owner.post(endpoint, json={**body, "password": "неверный пароль 12345"})
        ).status_code == 403
        assert (
            await owner.post(endpoint, json={**body, "confirmation_name": "wrong"})
        ).status_code == 422

        key = app.state.settings.encryption_key
        assert key is not None
        legacy_payload = (
            Fernet(key.get_secret_value().encode())
            .encrypt(json.dumps({"subject": "Приглашение в Dialogentis", "text": "old"}).encode())
            .decode()
        )
        async with unit_of_work(app.state.session_factory) as db:
            legacy = EmailOutbox(encrypted_payload=legacy_payload)
            db.add(legacy)
            await db.flush()
            legacy_id = legacy.id
        assert (await owner.post(endpoint, json=body)).status_code == 503
        async with unit_of_work(app.state.session_factory) as db:
            await db.execute(delete(EmailOutbox).where(EmailOutbox.id == legacy_id))

        one, two = await asyncio.gather(
            owner.post(endpoint, json=body), owner2.post(endpoint, json=body)
        )
        assert one.status_code == two.status_code == 202, (one.text, two.text)
        job_id = UUID(one.json()["job_id"])
        assert UUID(two.json()["job_id"]) == job_id
        assert (await owner.get(f"/api/v1/organizations/{org_id}")).status_code == 404
        assert (await guest.get(f"/api/v1/organizations/{org_id}")).status_code == 404
        assert (await guest.get(f"/api/v1/organizations/{other_id}")).status_code == 200
        assert (
            await pending.post("/api/v1/invitations/accept", json={"token": pending_token})
        ).status_code == 400
        assert not await cancel(app.state.session_factory, job_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert len((await db.scalars(select(OrganizationDeletion))).all()) == 1
            assert (
                len(
                    (
                        await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
                    ).all()
                )
                == 1
            )
            job = await db.get(Job, job_id)
            assert job is not None
            job.status = "failed"
            job.error_code = "temporary_error"
        retry = f"/api/v1/deletions/{job_id}/retry"
        assert (await guest.post(retry, json={"password": PASSWORD})).status_code == 404
        assert (
            await owner.post(retry, json={"password": "неверный пароль 12345"})
        ).status_code == 403
        assert (await owner.post(retry, json={"password": PASSWORD})).json()["status"] == "queued"
        assert (await owner.post(retry, json={"password": PASSWORD})).json()["status"] == "queued"
        await process_job(app.state.session_factory, job_id, org_id)
        assert not (artifact_root / str(org_id)).exists()
        assert storage.read(other_id, other_file_id, other_info, max_bytes=100) == b"keep me"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            events = (
                await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
            ).all()
            assert len(events) == 2
            for event in events:
                event.published_at = datetime.now(UTC)
        status = await owner.get(f"/api/v1/deletions/{job_id}")
        assert status.status_code == 200 and status.json()["status"] == "succeeded"
        assert (await guest.get(f"/api/v1/deletions/{job_id}")).status_code == 404
        assert read_markers(tmp_path) == [org_id]
        assert "Удаляемая" not in (tmp_path / f"{org_id}.json").read_text()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(Organization, org_id) is None
            assert (await db.scalars(select(Agent))).all() == []
            assert (await db.scalars(select(AgentDraft))).all() == []
            assert (
                await db.scalars(select(Membership).where(Membership.organization_id == org_id))
            ).all() == []
            record = await db.get(OrganizationDeletion, job_id)
            assert record is not None and record.completed_at is not None
            assert (
                await db.scalars(select(EmailOutbox).where(EmailOutbox.organization_id == org_id))
            ).all() == []
        async with unit_of_work(app.state.session_factory, organization_id=other_id) as db:
            assert await db.get(Agent, other_agent_id) is not None
            assert (
                await db.get(User, owner_id) is not None
                and await db.get(User, guest_id) is not None
            )

        # Recreate the old tenant as if an earlier backup had been restored.
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Organization(id=org_id, name="Restored", slug=f"org-{uuid4()}"))
            await db.flush()
            db.add(Membership(organization_id=org_id, user_id=owner_id, role="owner"))
            db.add(Agent(id=uuid4(), organization_id=org_id, name="Old", slug="old"))
        storage.write(org_id, file_id, [b"restored"], max_bytes=100)
        assert await replay(app.state.session_factory, tmp_path, key.get_secret_value()) == 1
        assert not (artifact_root / str(org_id)).exists()
        assert await replay(app.state.session_factory, tmp_path, key.get_secret_value()) == 1
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(Organization, org_id) is None
            assert (await db.scalars(select(Agent))).all() == []
        async with unit_of_work(app.state.session_factory, organization_id=other_id) as db:
            assert await db.get(Agent, other_agent_id) is not None
        old_id = uuid4()
        async with unit_of_work(app.state.session_factory, organization_id=old_id) as db:
            db.add(Organization(id=old_id, name="Old backup", slug=f"org-{uuid4()}"))
            await db.flush()
            db.add(Agent(id=uuid4(), organization_id=old_id, name="Old agent", slug="old"))
        write_marker(tmp_path, old_id)
        assert await replay(app.state.session_factory, tmp_path, key.get_secret_value()) == 2
        async with unit_of_work(app.state.session_factory, organization_id=old_id) as db:
            assert await db.get(Organization, old_id) is None
            assert (await db.scalars(select(Agent))).all() == []
    get_settings.cache_clear()


async def test_journal_failure_retries_without_deleting_data(app, tmp_path, monkeypatch):
    from dialogentis.jobs.tasks import process_job

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked")
    monkeypatch.setenv("DELETION_JOURNAL_ROOT", str(blocker))
    get_settings.cache_clear()
    async with client_for(app) as owner:
        await account(app, owner)
        response = await owner.post(
            "/api/v1/organizations", json={"name": "Сохранить до повтора", "slug": f"org-{uuid4()}"}
        )
        org_id = UUID(response.json()["id"])
        result = await owner.post(
            f"/api/v1/organizations/{org_id}/deletion",
            json={"password": PASSWORD, "confirmation_name": "Сохранить до повтора"},
        )
        assert result.status_code == 202
        job_id = UUID(result.json()["job_id"])
        await process_job(app.state.session_factory, job_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org, job = await db.get(Organization, org_id), await db.get(Job, job_id)
            assert org is not None and job is not None
            assert job.status == "queued" and job.error_code == "temporary_error"
            job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
        monkeypatch.setenv("DELETION_JOURNAL_ROOT", str(tmp_path / "journal"))
        get_settings.cache_clear()
        await process_job(app.state.session_factory, job_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(Organization, org_id) is None
            job = await db.get(Job, job_id)
            assert job is not None and job.status == "succeeded" and job.attempts == 2
            for event in (
                await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
            ).all():
                event.published_at = datetime.now(UTC)


async def test_artifact_failure_retries_without_deleting_database(app, tmp_path, monkeypatch):
    from dialogentis.jobs.tasks import process_job

    monkeypatch.setenv("DELETION_JOURNAL_ROOT", str(tmp_path / "journal"))
    artifact_root = tmp_path / "unavailable"
    monkeypatch.setenv("ARTIFACT_ROOT", str(artifact_root))
    get_settings.cache_clear()
    async with client_for(app) as owner:
        await account(app, owner)
        response = await owner.post(
            "/api/v1/organizations",
            json={"name": "Удалить после повтора", "slug": f"org-{uuid4()}"},
        )
        org_id = UUID(response.json()["id"])
        result = await owner.post(
            f"/api/v1/organizations/{org_id}/deletion",
            json={"password": PASSWORD, "confirmation_name": "Удалить после повтора"},
        )
        job_id = UUID(result.json()["job_id"])
        await process_job(app.state.session_factory, job_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org, job = await db.get(Organization, org_id), await db.get(Job, job_id)
            assert org is not None and job is not None
            assert job.status == "queued" and job.error_code == "temporary_error"
            job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)
        artifact_root.mkdir()
        await process_job(app.state.session_factory, job_id, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(Organization, org_id) is None
            job = await db.get(Job, job_id)
            assert job is not None and job.status == "succeeded"
            for event in (
                await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
            ).all():
                event.published_at = datetime.now(UTC)


def test_journal_rejects_corruption(tmp_path):
    org_id = uuid4()
    write_marker(tmp_path, org_id)
    assert read_markers(tmp_path) == [org_id]
    (tmp_path / f"{org_id}.json").write_text('{"version":1}')
    with pytest.raises(ValueError):
        read_markers(tmp_path)
