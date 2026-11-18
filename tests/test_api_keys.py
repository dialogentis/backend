import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

from dialogentis.auth.security import digest
from dialogentis.channels.api_key_models import AgentApiKey
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.models import Membership
from dialogentis.organizations import AuditEvent
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_api_keys_are_agent_scoped_and_secret_is_returned_once(app):
    factory = app.state.session_factory
    async with client_for(app) as owner, client_for(app) as member, client_for(app) as anonymous:
        owner_id, org_id, agent_id = await create_agent(app, owner)
        member_id, other_org, other_agent = await create_agent(app, member)
        agents = f"/api/v1/organizations/{org_id}/agents"
        second = await owner.post(agents, json={"name": "Second", "slug": "second"})
        assert second.status_code == 201
        path = f"{agents}/{agent_id}/api-keys"
        wrong_agent = f"{agents}/{second.json()['id']}/api-keys"
        wrong_tenant = f"/api/v1/organizations/{other_org}/agents/{other_agent}/api-keys"
        data = {"name": "  Connector  "}

        assert (await member.get(path)).status_code == 404
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=member_id, role="editor"))
        assert (await member.get(path)).status_code == 403
        assert (await member.post(path, json=data)).status_code == 403
        assert (await owner.post(path, json={"name": "  "})).status_code == 422
        now = datetime.now(UTC)
        for expiry in (now - timedelta(days=1), now + timedelta(days=366)):
            assert (
                await owner.post(path, json={**data, "expires_at": expiry.isoformat()})
            ).status_code == 422
        response = await owner.post(path, json=data)
        assert response.status_code == 201, response.text
        assert response.headers.get("cache-control") == "no-store"
        issued = response.json()
        key_id = UUID(issued["id"])
        secret = issued.pop("secret")
        assert issued["name"] == "Connector" and issued["prefix"] == secret[:12]
        assert issued["scopes"] == ["conversation:create", "message:send"]
        assert issued["created_by"] == str(owner_id)
        assert datetime.fromisoformat(issued["expires_at"]) - datetime.fromisoformat(
            issued["created_at"]
        ) == timedelta(days=90)
        assert (await owner.get(path)).json()["items"] == [issued]
        assert (await owner.get(wrong_agent)).json()["items"] == []
        assert (await member.get(wrong_tenant)).json()["items"] == []
        assert (
            await anonymous.get(path, headers={"Authorization": f"Bearer {secret}"})
        ).status_code == 401

        async with unit_of_work(factory, organization_id=org_id) as db:
            key = await db.get(AgentApiKey, key_id)
            assert key is not None and key.token_hash == digest(secret)
            with pytest.raises(DBAPIError):
                async with db.begin_nested():
                    await db.execute(
                        update(AgentApiKey).where(AgentApiKey.id == key_id).values(token_hash="x")
                    )
            membership = await db.get(Membership, (org_id, member_id))
            assert membership is not None
            membership.role = "admin"
        async with unit_of_work(factory, organization_id=other_org) as db:
            assert await db.get(AgentApiKey, key_id) is None
        async with unit_of_work(factory) as db:
            assert await db.get(AgentApiKey, key_id) is None

        expiry = now + timedelta(days=1)
        second_key = await member.post(
            path, json={"name": "Short", "expires_at": expiry.isoformat()}
        )
        assert second_key.status_code == 201, second_key.text
        assert datetime.fromisoformat(second_key.json()["expires_at"]) == expiry
        first_page = (await owner.get(path, params={"limit": 1})).json()
        assert first_page["items"][0]["id"] == second_key.json()["id"]
        assert "secret" not in first_page["items"][0]
        assert (
            await owner.get(path, params={"limit": 1, "cursor": first_page["next_cursor"]})
        ).json() == {"items": [issued], "next_cursor": None}
        revoke_path = f"/{key_id}/revoke"
        assert (await owner.post(wrong_agent + revoke_path)).status_code == 404
        assert (await member.post(wrong_tenant + revoke_path)).status_code == 404
        for _ in range(2):
            assert (await member.post(path + revoke_path)).status_code == 204
        revoked = (await owner.get(path)).json()["items"][1]
        assert revoked["revoked_at"] is not None
        assert {**revoked, "revoked_at": None} == issued
        async with unit_of_work(factory, organization_id=org_id) as db:
            events = (
                await db.scalars(select(AuditEvent).where(AuditEvent.target_id == key_id))
            ).all()
            assert {event.action for event in events} == {"api_key.created", "api_key.revoked"}
            assert len(events) == 2
            assert all(event.details == {"agent_id": str(agent_id)} for event in events)
            await purge_tenant(db, org_id)
        async with unit_of_work(factory, organization_id=org_id) as db:
            assert await db.get(AgentApiKey, key_id) is None
