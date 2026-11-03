import asyncio
import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.models import AgentDraft
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


async def test_agent_metadata_roles_isolation_and_optimistic_concurrency(app):
    async with client_for(app) as owner, client_for(app) as owner2, client_for(app) as member:
        owner_email, _ = await account(app, owner)
        member_email, member_id = await account(app, member)
        login = await owner2.post(
            "/api/v1/auth/login", json={"email": owner_email, "password": PASSWORD}
        )
        assert login.status_code == 200
        owner2.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        first = await owner.post(
            "/api/v1/organizations", json={"name": "First", "slug": f"org-{uuid4()}"}
        )
        second = await member.post(
            "/api/v1/organizations", json={"name": "Second", "slug": f"org-{uuid4()}"}
        )
        org_id, other_id = first.json()["id"], second.json()["id"]
        await owner.post(
            f"/api/v1/organizations/{org_id}/invitations",
            json={"email": member_email, "role": "viewer"},
        )
        token = await mail_token(app, member_email, "Приглашение")
        assert (
            await member.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 200
        base = f"/api/v1/organizations/{org_id}/agents"
        body = {"name": " Помощник ", "slug": "support", "description": "Поддержка"}
        assert (await member.post(base, json=body)).status_code == 403
        created = await owner.post(base, json=body)
        assert created.status_code == 201, created.text
        assert created.json()["name"] == "Помощник"
        assert created.headers["etag"] == '"rev-1"'
        agent_id = created.json()["id"]
        async with unit_of_work(app.state.session_factory, organization_id=UUID(org_id)) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == UUID(agent_id)))
            assert draft is not None and draft.revision == 1
            assert {node["type"] for node in draft.configuration["flow"]["nodes"]} == {
                "start",
                "switch_intent",
                "fallback",
                "end",
            }
        assert (await owner.post(base, json=body)).status_code == 409
        assert (await owner.post(base, json={**body, "name": "   "})).status_code == 422
        assert (await member.get(f"{base}/{agent_id}")).status_code == 200
        assert (
            await member.post(f"{base}/{agent_id}/copy", json={"name": "Copy", "slug": "copy"})
        ).status_code == 403
        assert (
            await owner.get(f"/api/v1/organizations/{other_id}/agents/{agent_id}")
        ).status_code == 404
        assert (
            await member.get(f"/api/v1/organizations/{other_id}/agents/{agent_id}")
        ).status_code == 404
        assert (
            await member.patch(
                f"{base}/{agent_id}", json={"name": "X"}, headers={"If-Match": '"rev-1"'}
            )
        ).status_code == 403
        assert (await owner.patch(f"{base}/{agent_id}", json={"name": "X"})).status_code == 422
        assert (
            await owner.patch(
                f"{base}/{agent_id}", json={"name": None}, headers={"If-Match": '"rev-1"'}
            )
        ).status_code == 422
        one, two = await asyncio.gather(
            owner.patch(
                f"{base}/{agent_id}", json={"name": "One"}, headers={"If-Match": '"rev-1"'}
            ),
            owner2.patch(
                f"{base}/{agent_id}", json={"name": "Two"}, headers={"If-Match": '"rev-1"'}
            ),
        )
        assert sorted([one.status_code, two.status_code]) == [200, 412]
        assert (await owner.get(f"{base}/{agent_id}")).headers["etag"] == '"rev-2"'
        assert (
            await owner.patch(
                f"/api/v1/organizations/{org_id}/members/{member_id}", json={"role": "editor"}
            )
        ).status_code == 200
        assert (await member.post(f"{base}/{agent_id}/archive")).json()["status"] == "archived"
        assert (await member.post(f"{base}/{agent_id}/restore")).json()["status"] == "active"
        assert (await member.post(base, json={**body, "slug": "second"})).status_code == 201
        page1 = (await owner.get(base, params={"limit": 1})).json()
        assert len(page1["items"]) == 1 and page1["next_cursor"]
        page2 = (await owner.get(base, params={"limit": 1, "cursor": page1["next_cursor"]})).json()
        assert len(page2["items"]) == 1 and page2["next_cursor"] is None
        assert page1["items"][0]["id"] != page2["items"][0]["id"]
        assert (await owner.get(base, params={"limit": 101})).status_code == 422
        assert (
            len((await owner.get(base, params={"status": "active", "q": "second"})).json()["items"])
            == 1
        )
        assert (await owner.get(base, params={"q": "%"})).json()["items"] == []
        assert (await owner.get(base, params={"status": "archived"})).json()["items"] == []
        source = (await owner.get(f"{base}/{agent_id}/draft")).json()["config"]
        copied = await member.post(f"{base}/{agent_id}/copy", json={"name": "Copy", "slug": "copy"})
        assert copied.status_code == 201 and copied.json()["draft_revision"] == 1
        clone_id = copied.json()["id"]
        cloned = (await owner.get(f"{base}/{clone_id}/draft")).json()["config"]
        old_nodes = {node["id"]: node for node in source["flow"]["nodes"]}
        new_nodes = {node["id"]: node for node in cloned["flow"]["nodes"]}
        assert not set(old_nodes) & set(new_nodes)
        assert source["flow"]["flow_id"] != cloned["flow"]["flow_id"]
        assert {node["type"] for node in old_nodes.values()} == {
            node["type"] for node in new_nodes.values()
        }
        assert not {edge["id"] for edge in source["flow"]["edges"]} & {
            edge["id"] for edge in cloned["flow"]["edges"]
        }
        assert all(
            edge["source"] in new_nodes and edge["target"] in new_nodes
            for edge in cloned["flow"]["edges"]
        )
        fallback = next(node for node in new_nodes.values() if node["type"] == "fallback")
        assert new_nodes[fallback["config"]["resume_node_id"]]["type"] == "switch_intent"
        assert (
            await owner.post(f"{base}/{agent_id}/copy", json={"name": "Copy", "slug": "copy"})
        ).status_code == 409
        async with unit_of_work(app.state.session_factory, organization_id=UUID(org_id)) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == UUID(agent_id)))
            assert draft is not None
            draft.configuration = {
                **draft.configuration,
                "slots": [{"id": str(uuid4()), "code": "city"}],
            }
        assert (
            await owner.post(f"{base}/{agent_id}/copy", json={"name": "Unsafe", "slug": "unsafe"})
        ).status_code == 409
        assert len((await owner.get(base)).json()["items"]) == 3
