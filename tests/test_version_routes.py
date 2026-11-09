import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.db import unit_of_work
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import Membership
from dialogentis.versions.models import AgentVersion
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_version_api_training_reports_pagination_and_permissions(app):
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
        assert (await owner.get(f"{base}/{version_id}/report")).status_code == 409
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
        result = await viewer.get(f"{base}/{version_id}/report")
        assert result.status_code == 200, result.text
        report = result.json()
        assert report["version_id"] == version_id and report["training_examples"] == 10
        assert report["intents"]["test_count"] == 2 and report["graph_validation"] == "not_checked"
        assert len(report["intent_catalog"]) == 3
        assert "errors" not in report["intents"] and "errors" not in report["entities"]
        errors = await viewer.get(
            f"{base}/{version_id}/report/errors", params={"kind": "intent", "limit": 1}
        )
        assert errors.status_code == 200 and len(errors.json()["items"]) <= 1
        if errors.json()["items"]:
            assert isinstance(errors.json()["items"][0]["text"], str)
        assert (
            await viewer.get(
                f"{base}/{version_id}/report/errors", params={"kind": "entity", "limit": 101}
            )
        ).status_code == 422
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


async def test_invalid_version_request_does_not_enqueue(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}/versions"
        result = await owner.post(
            base, json={"source_revision": 1}, headers={"Idempotency-Key": "invalid-data"}
        )
        assert result.status_code == 422 and result.json()["error"]["details"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert (await db.scalars(select(AgentVersion))).all() == []


async def test_settings_are_saved_in_version_and_applied_to_report(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        await populate(app, org_id, agent_id)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        draft = (await owner.get(f"{base}/draft")).json()
        assert draft["config"]["nlu_settings"] == {"score": 0.55, "margin": 0.15, "similarity": 0.2}
        grant = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        headers = {
            "X-Edit-Lease": grant["lease_token"],
            "X-Edit-Fence": str(grant["fencing_counter"]),
            "If-Match": '"rev-1"',
        }
        body = {"schema_version": 1, "config": draft["config"]}
        settings = {"score": 0.73, "margin": 0.25, "similarity": 0.31}
        body["config"]["nlu_settings"] = settings
        saved = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert saved.status_code == 200 and saved.json()["config"]["nlu_settings"] == settings
        created = await owner.post(
            f"{base}/versions",
            json={"source_revision": 2},
            headers={"Idempotency-Key": "custom-settings"},
        )
        assert created.status_code == 202, created.text
        version_id, job_id = created.json()["version_id"], created.json()["job_id"]
        body["config"]["nlu_settings"] = {"score": 0.1, "margin": 0.1, "similarity": 0.1}
        saved = await owner.put(
            f"{base}/draft", headers={**headers, "If-Match": '"rev-2"'}, json=body
        )
        assert saved.status_code == 200
        await process_job(app.state.session_factory, UUID(job_id), org_id, HANDLERS, "training")
        report = await owner.get(f"{base}/versions/{version_id}/report")
        assert report.status_code == 200 and report.json()["nlu_settings"] == settings
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            version = await db.get(AgentVersion, UUID(version_id))
            assert version is not None and version.snapshot["nlu_settings"] == settings
