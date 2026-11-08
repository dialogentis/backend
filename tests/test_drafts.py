import asyncio
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any
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


def headers(grant):
    return {"X-Edit-Lease": grant["lease_token"], "X-Edit-Fence": str(grant["fencing_counter"])}


async def test_lease_fencing_permissions_and_atomic_draft_save(app):
    async with client_for(app) as owner, client_for(app) as owner2, client_for(app) as editor:
        owner_email, _ = await account(app, owner)
        editor_email, editor_id = await account(app, editor)
        login = await owner2.post(
            "/api/v1/auth/login", json={"email": owner_email, "password": PASSWORD}
        )
        owner2.headers["X-CSRF-Token"] = login.json()["csrf_token"]
        org = await owner.post(
            "/api/v1/organizations", json={"name": "Team", "slug": f"org-{uuid4()}"}
        )
        org_id = org.json()["id"]
        await owner.post(
            f"/api/v1/organizations/{org_id}/invitations",
            json={"email": editor_email, "role": "editor"},
        )
        token = await mail_token(app, editor_email, "Приглашение")
        assert (
            await editor.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 200
        agent = await owner.post(
            f"/api/v1/organizations/{org_id}/agents", json={"name": "Agent", "slug": "agent"}
        )
        agent_id = agent.json()["id"]
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        original = await owner.get(f"{base}/draft")
        assert original.status_code == 200 and original.headers["etag"] == '"rev-1"'
        assert original.json()["lease"] is None
        dataset = await editor.get(f"{base}/draft/dataset")
        assert dataset.status_code == 200 and dataset.json()["revision"] == 1
        assert dataset.json()["dataset"]["intents"] == []
        initial_validation = await owner.post(f"{base}/draft/validate", json={"revision": 1})
        assert initial_validation.status_code == 200
        assert (
            initial_validation.json()["scope"] == "dataset"
            and not initial_validation.json()["complete"]
        )
        assert "insufficient_intents" in {
            item["code"] for item in initial_validation.json()["errors"]
        }
        assert (await owner.post(f"{base}/draft/validate", json={"revision": 2})).status_code == 409
        body: dict[str, Any] = {"schema_version": 1, "config": original.json()["config"]}
        a, b = await asyncio.gather(
            owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())}),
            owner2.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())}),
        )
        assert sorted([a.status_code, b.status_code]) == [200, 423]
        holder, grant = (owner, a.json()) if a.status_code == 200 else (owner2, b.json())
        observed = (await editor.get(f"{base}/draft")).json()
        assert observed["lease"]["holder_name"].endswith("***")
        assert grant["lease_token"] not in str(observed)
        auth = headers(grant)
        assert (await holder.put(f"{base}/draft", json=body)).status_code == 422
        assert (
            await holder.put(f"{base}/draft", json=body, headers={**auth, "If-Match": '"rev-9"'})
        ).status_code == 412
        assert (
            await editor.put(f"{base}/draft", json=body, headers={**auth, "If-Match": '"rev-1"'})
        ).status_code == 423
        assert (
            await holder.put(
                f"{base}/draft",
                json={**body, "extra": True},
                headers={**auth, "If-Match": '"rev-1"'},
            )
        ).status_code == 422
        body["config"]["slots"].append({"id": str(uuid4()), "code": "city"})
        saved = await holder.put(
            f"{base}/draft", json=body, headers={**auth, "If-Match": '"rev-1"'}
        )
        assert saved.status_code == 200 and saved.headers["etag"] == '"rev-2"'
        assert saved.json()["config"]["slots"][0]["code"] == "city"
        assert (
            await holder.put(f"{base}/draft", json=body, headers={**auth, "If-Match": '"rev-1"'})
        ).status_code == 412
        assert (await holder.post(f"{base}/edit-lease/renew", headers=auth)).status_code == 200
        assert (await editor.post(f"{base}/edit-lease/revoke")).status_code == 403
        assert (await owner.post(f"{base}/edit-lease/revoke")).status_code == 204
        assert (
            await holder.put(f"{base}/draft", json=body, headers={**auth, "If-Match": '"rev-2"'})
        ).status_code == 423
        new = await editor.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})
        assert new.status_code == 200 and new.json()["fencing_counter"] > grant["fencing_counter"]
        editor_auth = headers(new.json())
        assert (
            await owner.patch(
                f"/api/v1/organizations/{org_id}/members/{editor_id}", json={"role": "viewer"}
            )
        ).status_code == 200
        assert (
            await editor.post(f"{base}/edit-lease/renew", headers=editor_auth)
        ).status_code == 403
        assert (
            await editor.post(f"{base}/draft/validate", json={"revision": 2})
        ).status_code == 403
        assert (
            await editor.put(
                f"{base}/draft", json=body, headers={**editor_auth, "If-Match": '"rev-2"'}
            )
        ).status_code == 403
        assert (await owner.post(f"{base}/edit-lease/revoke")).status_code == 204
        next_grant = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        async with unit_of_work(app.state.session_factory, organization_id=UUID(org_id)) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == UUID(agent_id)))
            assert draft is not None
            draft.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        renewed = await owner2.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})
        assert renewed.status_code == 200
        assert renewed.json()["fencing_counter"] > next_grant["fencing_counter"]
        assert (
            await owner.put(
                f"{base}/draft", json=body, headers={**headers(next_grant), "If-Match": '"rev-2"'}
            )
        ).status_code == 423
        assert (
            await owner2.delete(f"{base}/edit-lease", headers=headers(renewed.json()))
        ).status_code == 204
        assert (await owner.get(f"{base}/draft")).json()["lease"] is None
        final_grant = (await owner.post(f"{base}/edit-lease", json={"tab_id": str(uuid4())})).json()
        first, second = str(uuid4()), str(uuid4())
        body["config"]["intents"] = [
            {"id": first, "code": "first", "name": "Первое"},
            {"id": second, "code": "second", "name": "Второе"},
        ]
        body["config"]["entities"] = [
            {
                "id": str(uuid4()),
                "code": "bad_regex",
                "name": "Проверка",
                "value_type": "string",
                "method": "regex",
                "pattern": r"(a)\1",
            }
        ]
        body["config"]["examples"] = [
            {
                "id": str(uuid4()),
                "text": "PRIVATE_SENTINEL",
                "intent_id": first,
                "split": "train",
                "spans": [],
            },
            {
                "id": str(uuid4()),
                "text": "private_sentinel",
                "intent_id": first,
                "split": "test",
                "spans": [],
            },
        ]
        rejected = await owner.put(
            f"{base}/draft", json=body, headers={**headers(final_grant), "If-Match": '"rev-2"'}
        )
        assert rejected.status_code == 422
        assert {"duplicate_across_splits", "invalid_regex"} <= {
            item["code"] for item in rejected.json()["error"]["details"]
        }
        assert "PRIVATE_SENTINEL" not in rejected.text
        assert (await owner.get(f"{base}/draft")).json()["revision"] == 2
        # Existing invalid drafts remain readable and can be diagnosed and corrected.
        async with unit_of_work(app.state.session_factory, organization_id=UUID(org_id)) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == UUID(agent_id)))
            assert draft is not None
            draft.configuration = body["config"]
            draft.revision = 3
        validation = await owner.post(f"{base}/draft/validate", json={"revision": 3})
        assert validation.status_code == 200
        assert {"duplicate_across_splits", "invalid_regex"} <= {
            item["code"] for item in validation.json()["errors"]
        }
        assert "PRIVATE_SENTINEL" not in validation.text
        body["config"]["slots"] = []
        body["config"]["entities"][0]["pattern"] = r"\d+"
        body["config"]["examples"][1]["text"] = "Другая фраза"
        assert (
            await owner.put(
                f"{base}/draft", json=body, headers={**headers(final_grant), "If-Match": '"rev-3"'}
            )
        ).status_code == 200
        invalid_reference = deepcopy(body)
        invalid_reference["config"]["examples"][1]["intent_id"] = str(uuid4())
        bad = await owner.put(
            f"{base}/draft",
            json=invalid_reference,
            headers={**headers(final_grant), "If-Match": '"rev-4"'},
        )
        assert bad.status_code == 422
        assert any(issue["code"] == "unknown_intent" for issue in bad.json()["error"]["details"])
        coerced_boolean = deepcopy(body)
        coerced_boolean["config"]["intents"][0]["enabled"] = "yes"
        assert (
            await owner.put(
                f"{base}/draft",
                json=coerced_boolean,
                headers={**headers(final_grant), "If-Match": '"rev-4"'},
            )
        ).status_code == 422
        assert (await owner.get(f"{base}/draft")).json()["revision"] == 4
        dataset = await owner.get(f"{base}/draft/dataset")
        assert dataset.status_code == 200 and dataset.json()["revision"] == 4
        assert dataset.json()["dataset"]["intents"][0]["enabled"] is True
        entity_id = body["config"]["entities"][0]["id"]
        preview_path = f"{base}/draft/entities/{entity_id}/preview"
        preview = await owner.post(preview_path, json={"revision": 4, "text": "📦 123"})
        assert preview.status_code == 200
        assert preview.json()["matches"][0]["text"] == "123"
        assert preview.json()["matches"][0]["start"] == 2
        assert (
            await owner.post(preview_path, json={"revision": 3, "text": "123"})
        ).status_code == 409
        assert (
            await editor.post(preview_path, json={"revision": 4, "text": "123"})
        ).status_code == 403
        assert (
            await owner.post(
                f"{base}/draft/entities/{uuid4()}/preview", json={"revision": 4, "text": "123"}
            )
        ).status_code == 404
        unknown_field = deepcopy(body)
        unknown_field["config"]["intents"][0]["PRIVATE_FIELD"] = "PRIVATE_VALUE"
        bad = await owner.put(
            f"{base}/draft",
            json=unknown_field,
            headers={**headers(final_grant), "If-Match": '"rev-4"'},
        )
        assert (
            bad.status_code == 422
            and "PRIVATE_FIELD" not in bad.text
            and "PRIVATE_VALUE" not in bad.text
        )
        assert any("<unknown>" in item["path"] for item in bad.json()["error"]["details"])
        copied = await owner.post(f"{base}/copy", json={"name": "Copy", "slug": "copy"})
        assert copied.status_code == 201
        clone_id = copied.json()["id"]
        clone = (await owner.get(f"/api/v1/organizations/{org_id}/agents/{clone_id}/draft")).json()[
            "config"
        ]
        assert clone["intents"][0]["id"] != first
        assert clone["examples"][0]["intent_id"] == clone["intents"][0]["id"]
        assert clone["examples"][0]["text"] == "PRIVATE_SENTINEL"
        async with unit_of_work(app.state.session_factory, organization_id=UUID(org_id)) as db:
            draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == UUID(agent_id)))
            assert draft is not None
            draft.configuration = unknown_field["config"]
        assert (await owner.get(f"{base}/draft")).status_code == 200
        invalid_dataset = await owner.get(f"{base}/draft/dataset")
        assert invalid_dataset.status_code == 422 and "PRIVATE_FIELD" not in invalid_dataset.text
