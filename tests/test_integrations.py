import os
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.integrations.models import Integration
from dialogentis.integrations.secrets import open_secret
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.models import AgentDraft, Membership
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_integrations_permissions_rotation_isolation_and_references(app):
    async with client_for(app) as owner, client_for(app) as viewer, client_for(app) as stranger:
        owner_id, org_id, agent_id = await create_agent(app, owner)
        viewer_id, _, _ = await create_agent(app, viewer)
        _, other_org, _ = await create_agent(app, stranger)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        path = f"/api/v1/organizations/{org_id}/integrations"
        data = {
            "name": "Orders",
            "base_url": "https://orders.example.com/api",
            "methods": ["GET", "POST"],
            "paths": ["/orders/"],
            "response_fields": ["status"],
            "secret": "private-token",
        }
        assert (await viewer.post(path, json=data)).status_code == 403
        assert (await stranger.post(path, json=data)).status_code == 404
        created = await owner.post(path, json=data)
        assert created.status_code == 201, created.text
        assert (await owner.post(path, json=data)).status_code == 409
        spare = await owner.post(path, json={**data, "name": "Unused"})
        assert spare.status_code == 201
        spare_path = f"{path}/{spare.json()['id']}"
        assert (
            await owner.delete(spare_path, headers={"If-Match": spare.headers["etag"]})
        ).status_code == 204
        assert (await owner.get(spare_path)).status_code == 404
        body, etag = created.json(), created.headers["etag"]
        integration_id = UUID(body["id"])
        assert body["has_secret"] and "private-token" not in created.text and "secret" not in body
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            item = await db.get(Integration, integration_id)
            assert item is not None and item.encrypted_secret is not None
            assert b"private-token" not in item.encrypted_secret
            assert (
                open_secret(item.encrypted_secret, org_id, integration_id, app.state.settings)
                == "private-token"
            )
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(Integration, integration_id) is None
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            with pytest.raises(DBAPIError):
                async with dispatcher() as db:
                    await db.execute(select(Integration))
        finally:
            await dispatcher.kw["bind"].dispose()
        item_path = f"{path}/{integration_id}"
        assert (await viewer.get(item_path)).status_code == 200
        assert (await stranger.get(item_path)).status_code == 404
        assert (
            await viewer.patch(item_path, json={"enabled": False}, headers={"If-Match": etag})
        ).status_code == 403
        assert (await owner.patch(item_path, json={"enabled": False})).status_code == 428
        changed = await owner.patch(item_path, json={"enabled": False}, headers={"If-Match": etag})
        assert changed.status_code == 200 and not changed.json()["enabled"]
        assert (
            await owner.patch(item_path, json={"enabled": True}, headers={"If-Match": etag})
        ).status_code == 412
        rotated = await owner.post(
            f"{item_path}/rotate-secret",
            json={"secret": "new-token"},
            headers={"If-Match": changed.headers["etag"]},
        )
        assert rotated.status_code == 200 and "new-token" not in rotated.text
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            item = await db.get(Integration, integration_id)
            assert item is not None and item.encrypted_secret is not None
            assert (
                open_secret(item.encrypted_secret, org_id, integration_id, app.state.settings)
                == "new-token"
            )
        await populate(app, org_id, agent_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            config["flow"]["nodes"].append(
                {
                    "id": str(uuid4()),
                    "type": "http_action",
                    "config": {"integration_id": f"URN:UUID:{integration_id}".upper()},
                    "position": {"x": 0, "y": 0},
                }
            )
            draft.configuration = config
        delete_headers = {"If-Match": rotated.headers["etag"]}
        assert (await owner.delete(item_path, headers=delete_headers)).status_code == 409
        await prepare(app, owner_id, org_id, agent_id, 1, "integration-reference")
        await mark_published(app, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
            assert draft is not None
            config = deepcopy(draft.configuration)
            config["flow"]["nodes"] = [
                node for node in config["flow"]["nodes"] if node["type"] != "http_action"
            ]
            draft.configuration = config
        assert (await owner.delete(item_path, headers=delete_headers)).status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.get(Integration, integration_id) is None
