import asyncio
import os
from copy import deepcopy
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from dialogentis.api_errors import DomainValidationError
from dialogentis.artifact_service import reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.jobs.lifecycle import cancel
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.models import Agent, AgentDraft, Membership, User
from dialogentis.nlu.settings import NLUSettings
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import TrainingInProgress, prepare_version, version_status
from tests.test_auth import account, client_for
from tests.test_exports import app as app
from tests.test_nlu_intents import training_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def create_agent(app, client):
    _, user_id = await account(app, client)
    response = await client.post(
        "/api/v1/organizations", json={"name": "Team", "slug": f"org-{uuid4()}"}
    )
    org_id = UUID(response.json()["id"])
    response = await client.post(
        f"/api/v1/organizations/{org_id}/agents", json={"name": "Bot", "slug": "bot"}
    )
    return UUID(user_id), org_id, UUID(response.json()["id"])


async def prepare(app, user_id, org_id, agent_id, revision, key):
    async with unit_of_work(
        app.state.session_factory, organization_id=org_id, user_id=user_id
    ) as db:
        user = await db.get(User, user_id)
        assert user is not None
        return await prepare_version(db, user, org_id, agent_id, revision, key)


async def populate(app, org_id, agent_id):
    async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
        draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
        assert draft is not None
        config = deepcopy(draft.configuration)
        config.update(training_dataset().model_dump(mode="json"))
        draft.configuration = config


async def test_snapshot_idempotency_concurrency_and_job_lifecycle(app):
    async with client_for(app) as client:
        user_id, org_id, agent_id = await create_agent(app, client)
        await populate(app, org_id, agent_id)
        first, repeated = await asyncio.gather(
            prepare(app, user_id, org_id, agent_id, 1, "same-key"),
            prepare(app, user_id, org_id, agent_id, 1, "same-key"),
        )
        assert first.version.id == repeated.version.id
        assert {first.replayed, repeated.replayed} == {False, True}
        version_id, job_id = first.version.id, first.version.job_id
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            version = await db.get(AgentVersion, version_id)
            job = await db.get(Job, job_id)
            event = await db.scalar(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
            assert version is not None and job is not None and event is not None
            assert version_status(version, job) == "building"
            incomplete = Job(
                id=job.id, organization_id=org_id, kind="agent.training", status="succeeded"
            )
            assert version_status(version, incomplete) == "failed"
            with pytest.raises(ValueError, match="does not belong"):
                version_status(
                    version,
                    Job(id=uuid4(), organization_id=org_id, kind="agent.training", status="queued"),
                )
            assert version.number == 1 and version.source_revision == 1
            assert version.snapshot["nlu_settings"] == NLUSettings().model_dump()
            assert version.idempotency_key_hash != "same-key"
            assert job.kind == "agent.training" and job.queue == event.queue == "training"
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            changed = deepcopy(draft.configuration)
            changed["flow"]["nodes"][0]["position"]["x"] = 123
            draft.configuration, draft.revision = changed, 2
        replayed = await prepare(app, user_id, org_id, agent_id, 1, "same-key")
        assert replayed.replayed and replayed.version.id == version_id
        assert replayed.version.snapshot["config"]["flow"]["nodes"][0]["position"]["x"] != 123
        with pytest.raises(HTTPException) as conflict:
            await prepare(app, user_id, org_id, agent_id, 2, "same-key")
        assert conflict.value.status_code == 409
        with pytest.raises(TrainingInProgress) as running:
            await prepare(app, user_id, org_id, agent_id, 2, "different-key")
        assert running.value.job_id == job_id
        assert await cancel(app.state.session_factory, job_id, org_id)
        outcomes = await asyncio.gather(
            prepare(app, user_id, org_id, agent_id, 2, "second-key"),
            prepare(app, user_id, org_id, agent_id, 2, "third-key"),
            return_exceptions=True,
        )
        assert sum(isinstance(item, TrainingInProgress) for item in outcomes) == 1
        created = next(item for item in outcomes if not isinstance(item, BaseException))
        assert created.version.number == 2 and created.version.source_revision == 2
        assert created.version.nlu_hash == first.version.nlu_hash
        assert created.version.config_hash != first.version.config_hash


async def test_snapshots_are_immutable_tenant_scoped_and_removed_with_organization(app):
    async with client_for(app) as owner, client_for(app) as stranger:
        user_id, org_id, agent_id = await create_agent(app, owner)
        other_id, other_org, other_agent = await create_agent(app, stranger)
        await populate(app, org_id, agent_id)
        result = await prepare(app, user_id, org_id, agent_id, 1, "version-key")
        version_id = result.version.id
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(AgentVersion, version_id) is None
        with pytest.raises(HTTPException) as forbidden:
            await prepare(app, other_id, org_id, agent_id, 1, "outside")
        assert forbidden.value.status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=other_id, role="viewer"))
        with pytest.raises(HTTPException) as viewer:
            await prepare(app, other_id, org_id, agent_id, 1, "viewer")
        assert viewer.value.status_code == 403
        with pytest.raises(IntegrityError):
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                await db.execute(
                    update(AgentVersion)
                    .where(AgentVersion.id == version_id)
                    .values(snapshot={"altered": True})
                )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            temporary = await reserve_artifact(
                db, org_id, kind="temporary", max_bytes=5, settings=app.state.settings
            )
            await write_artifact(
                db,
                ArtifactStorage(app.state.settings.artifact_root),
                org_id,
                temporary.id,
                [b"data"],
            )
            temporary_id = temporary.id
        with pytest.raises(IntegrityError):
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                version = await db.get(AgentVersion, version_id)
                assert version is not None
                version.model_artifact_id, version.validation_report = temporary_id, {}
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            model = await reserve_artifact(
                db, org_id, kind="model", max_bytes=5, settings=app.state.settings
            )
            await write_artifact(
                db, ArtifactStorage(app.state.settings.artifact_root), org_id, model.id, [b"model"]
            )
            version = await db.get(AgentVersion, version_id)
            assert version is not None
            version.model_artifact_id, version.validation_report = model.id, {"result": "stored"}
            job = Job(
                id=version.job_id, organization_id=org_id, kind="agent.training", status="succeeded"
            )
            assert version_status(version, job) == "ready"
        with pytest.raises(IntegrityError):
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                version = await db.get(AgentVersion, version_id)
                assert version is not None
                version.validation_report = {"result": "changed"}
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(AgentVersion, version_id) is None
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(Agent, other_agent) is not None


async def test_invalid_dataset_does_not_create_a_job(app):
    async with client_for(app) as client:
        user_id, org_id, agent_id = await create_agent(app, client)
        with pytest.raises(DomainValidationError):
            await prepare(app, user_id, org_id, agent_id, 1, "invalid-data")
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert (await db.scalars(select(AgentVersion))).all() == []
            assert (await db.scalars(select(Job).where(Job.kind == "agent.training"))).all() == []
