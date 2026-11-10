import os
from uuid import uuid4

import pytest

from dialogentis.db import unit_of_work
from dialogentis.models import Membership
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_structural_validation_is_revision_bound_and_tenant_scoped(app):
    async with client_for(app) as owner, client_for(app) as viewer:
        _, org_id, agent_id = await create_agent(app, owner)
        viewer_id, other_org, _ = await create_agent(app, viewer)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}/draft/flow/validate"
        result = await owner.post(base, json={"revision": 1})
        assert result.status_code == 200
        assert result.json()["scope"] == "flow_structure" and not result.json()["complete"]
        assert result.json()["errors"] == []
        assert {issue["code"] for issue in result.json()["warnings"]} == {"unreachable_node"}
        assert (await owner.post(base, json={"revision": 2})).status_code == 409
        assert (await owner.post(base, json={"revision": True})).status_code == 422
        assert (await viewer.post(base, json={"revision": 1})).status_code == 404
        wrong_tenant = f"/api/v1/organizations/{other_org}/agents/{agent_id}/draft/flow/validate"
        assert (await viewer.post(wrong_tenant, json={"revision": 1})).status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        assert (await viewer.post(base, json={"revision": 1})).status_code == 200
        assert (
            await owner.post(base.replace(str(agent_id), str(uuid4())), json={"revision": 1})
        ).status_code == 404


async def test_saved_intent_branch_cannot_dangle_after_direct_api_edit(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        draft = (await owner.get(f"{base}/draft")).json()
        grant = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        headers = {
            "X-Edit-Lease": grant["lease_token"],
            "X-Edit-Fence": str(grant["fencing_counter"]),
            "If-Match": '"rev-1"',
        }
        config = draft["config"]
        intent_id = str(uuid4())
        config["intents"] = [{"id": intent_id, "code": "order", "name": "Заказ"}]
        switch, end = config["flow"]["nodes"][1], config["flow"]["nodes"][3]
        config["flow"]["edges"].append(
            {
                "id": str(uuid4()),
                "source": switch["id"],
                "source_handle": intent_id,
                "target": end["id"],
            }
        )
        body = {"schema_version": 1, "config": config}
        saved = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert saved.status_code == 200, saved.text
        assert (await owner.post(f"{base}/draft/flow/validate", json={"revision": 2})).json()[
            "errors"
        ] == []
        config["intents"] = []
        headers["If-Match"] = '"rev-2"'
        rejected = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert rejected.status_code == 422
        assert "unknown_intent_output" in {
            item["code"] for item in rejected.json()["error"]["details"]
        }
        config["flow"]["edges"].pop()
        assert (await owner.put(f"{base}/draft", headers=headers, json=body)).status_code == 200


async def test_flow_validation_checks_saved_template_and_slot_references(app):
    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        draft = (await owner.get(f"{base}/draft")).json()
        grant = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        headers = {
            "X-Edit-Lease": grant["lease_token"],
            "X-Edit-Fence": str(grant["fencing_counter"]),
            "If-Match": '"rev-1"',
        }
        config = draft["config"]
        config["flow"]["nodes"][2]["config"]["text"] = "Город: {{city}}"
        body = {"schema_version": 1, "config": config}
        assert (await owner.put(f"{base}/draft", headers=headers, json=body)).status_code == 200
        validation = await owner.post(f"{base}/draft/flow/validate", json={"revision": 2})
        assert "unknown_template_slot" in {item["code"] for item in validation.json()["errors"]}
        config["slots"] = [{"id": str(uuid4()), "code": "city"}]
        headers["If-Match"] = '"rev-2"'
        assert (await owner.put(f"{base}/draft", headers=headers, json=body)).status_code == 200
        validation = await owner.post(f"{base}/draft/flow/validate", json={"revision": 3})
        assert validation.json()["errors"] == []
