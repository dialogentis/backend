import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.agents import empty_config
from dialogentis.artifact_service import reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.db import unit_of_work
from dialogentis.jobs.models import Job
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import Membership
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import canonical_hash
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_version_api_preparation_pagination_and_permissions(app):
    async with client_for(app) as owner, client_for(app) as viewer:
        _, org_id, agent_id = await create_agent(app, owner)
        viewer_id, other_org, _ = await create_agent(app, viewer)
        await populate(app, org_id, agent_id)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}/versions"
        assert (await owner.post(base, json={"source_revision": 1})).status_code == 422
        assert (
            await owner.post(
                base, json={"source_revision": True}, headers={"Idempotency-Key": "valid"}
            )
        ).status_code == 422
        assert (
            await owner.post(
                base, json={"source_revision": 1}, headers={"Idempotency-Key": "bad key"}
            )
        ).status_code == 422
        response = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "first-key"}
        )
        assert response.status_code == 202, response.text
        version_id, job_id = response.json()["version_id"], response.json()["job_id"]
        assert not response.json()["replayed"]
        repeated = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "first-key"}
        )
        assert repeated.status_code == 202 and repeated.json()["replayed"]
        assert repeated.json()["version_id"] == version_id
        conflict = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "second-key"}
        )
        assert (
            conflict.status_code == 409
            and conflict.json()["error"]["details"][0]["job_id"] == job_id
        )
        assert (await owner.get(f"{base}/{version_id}")).json()["status"] == "building"
        jobs = f"/api/v1/organizations/{org_id}/jobs/{job_id}"
        assert (await owner.get(jobs)).json()["version_id"] == version_id
        assert (await viewer.get(base)).status_code == 404
        assert (
            await viewer.get(
                f"/api/v1/organizations/{other_org}/agents/{agent_id}/versions/{version_id}"
            )
        ).status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        assert (await viewer.get(f"{base}/{version_id}")).status_code == 200
        assert (await viewer.get(jobs)).status_code == 403
        assert (await viewer.post(jobs + "/cancel")).status_code == 403
        await process_job(app.state.session_factory, UUID(job_id), org_id, HANDLERS, "training")
        result = await viewer.get(f"{base}/{version_id}")
        assert result.status_code == 200 and result.json()["status"] == "ready"
        assert result.json()["llm_model"] == "fixture"
        second = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "second-key"}
        )
        assert second.status_code == 202
        page = (await owner.get(base, params={"limit": 1})).json()
        assert page["items"][0]["number"] == 2 and page["next_cursor"] == "2"
        older = (await owner.get(base, params={"limit": 1, "cursor": page["next_cursor"]})).json()
        assert older["items"][0]["number"] == 1 and older["next_cursor"] is None
        second_job = second.json()["job_id"]
        cancel = await owner.post(f"/api/v1/organizations/{org_id}/jobs/{second_job}/cancel")
        assert cancel.status_code == 200 and cancel.json()["status"] == "cancelled"
        assert (await owner.get(f"{base}/{second.json()['version_id']}")).json()[
            "status"
        ] == "cancelled"


async def test_missing_llm_configuration_does_not_enqueue(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}/versions"
        app.state.settings.llm_model = ""
        result = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "invalid-data"}
        )
        assert result.status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert (await db.scalars(select(AgentVersion))).all() == []


async def test_obsolete_version_requires_rebuild_before_chat_or_publication(app):
    async with client_for(app) as owner:
        user_id, org_id, agent_id = await create_agent(app, owner)
        factory, settings = app.state.session_factory, app.state.settings
        version_id, job_id = uuid4(), uuid4()
        snapshot = {"schema_version": 1, "config": {**empty_config(), "nlu_settings": {}}}
        async with unit_of_work(factory, organization_id=org_id) as db:
            artifact = await reserve_artifact(
                db, org_id, kind="model", max_bytes=6, settings=settings
            )
            await write_artifact(
                db, ArtifactStorage(settings.artifact_root), org_id, artifact.id, [b"legacy"]
            )
            db.add(
                Job(
                    id=job_id,
                    organization_id=org_id,
                    kind="agent.training",
                    queue="training",
                    status="succeeded",
                    stage="evaluation",
                )
            )
            await db.flush()
            db.add(
                AgentVersion(
                    id=version_id,
                    organization_id=org_id,
                    agent_id=agent_id,
                    created_by=user_id,
                    number=1,
                    source_revision=1,
                    snapshot=snapshot,
                    config_hash=canonical_hash(snapshot),
                    nlu_hash="0" * 64,
                    idempotency_key_hash="1" * 64,
                    job_id=job_id,
                    model_artifact_id=artifact.id,
                    validation_report={},
                    dataset_warnings=[],
                )
            )

        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        version = await owner.get(f"{base}/versions/{version_id}")
        assert version.status_code == 200 and version.json()["status"] == "obsolete"
        assert version.json()["stage"] is None
        for mode in ("mock", "live"):
            started = await owner.post(
                f"{base}/test-conversations",
                json={"version_id": str(version_id), "http_mode": mode},
                headers={"Idempotency-Key": f"legacy-{mode}"},
            )
            assert started.status_code == 409
            assert "Подготовьте новую версию" in started.json()["error"]["message"]
        checks = await owner.get(f"{base}/versions/{version_id}/release-check")
        assert checks.json()["errors"][0]["code"] == "release_obsolete_version"

        created = await owner.post(
            f"{base}/versions",
            json={"source_revision": 1},
            headers={"Idempotency-Key": "refresh-version"},
        )
        assert created.status_code == 202
        await process_job(factory, UUID(created.json()["job_id"]), org_id, HANDLERS, "training")
        started = await owner.post(
            f"{base}/test-conversations", json={"version_id": created.json()["version_id"]}
        )
        assert started.status_code == 201
