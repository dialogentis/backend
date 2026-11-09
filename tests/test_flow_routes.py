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
