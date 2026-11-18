import asyncio
import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import AgentDraft, Membership
from dialogentis.organizations import AuditEvent
from dialogentis.versions.release_models import Release, ReleaseVersion
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_nlu_intents import training_dataset
from tests.test_runtime_handoff import call_flows
from tests.test_versions import create_agent, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_release_gates_atomic_publication_pins_and_rollback(app):
    async with client_for(app) as owner, client_for(app) as editor:
        owner_id, org_id, root_id = await create_agent(app, owner)
        editor_id, other_org, _ = await create_agent(app, editor)
        base = f"/api/v1/organizations/{org_id}/agents"
        root_url = f"{base}/{root_id}"
        target_response = await owner.post(base, json={"name": "Delivery", "slug": "delivery"})
        assert target_response.status_code == 201
        target_id = UUID(target_response.json()["id"])
        target_url = f"{base}/{target_id}"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=editor_id, role="editor"))

        root, target, _, _, handoff, _ = call_flows()

        async def train(agent_id, flow_config, revision, *, minimum=True):
            data = training_dataset().model_dump(mode="json")
            if minimum:
                data["examples"] = [
                    example for example in data["examples"] if example["split"] == "train"
                ]
                for intent, emoji in zip(data["intents"][:2], ("🛸", "🦄"), strict=True):
                    data["examples"].extend(
                        {
                            "id": str(uuid4()),
                            "intent_id": intent["id"],
                            "split": "test",
                            "text": emoji * repeat,
                            "expected_fallback": False,
                            "spans": [],
                        }
                        for repeat in range(1, 4)
                    )
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
                assert draft is not None
                draft.configuration = {
                    **draft.configuration,
                    **data,
                    "flow": flow_config["flow"],
                    "slots": flow_config["slots"],
                }
                draft.revision = revision
            prepared = await prepare(app, owner_id, org_id, agent_id, revision, uuid4().hex)
            await process_job(
                app.state.session_factory, prepared.version.job_id, org_id, HANDLERS, "training"
            )
            return prepared.version.id

        async def start(version_id):
            response = await owner.post(
                f"{root_url}/test-conversations",
                json={
                    "version_id": str(version_id),
                    "http_mode": "mock",
                },
            )
            assert response.status_code == 201, response.text
            return response.json()

        async def say(chat, text):
            prefix = f"/api/v1/organizations/{org_id}/test-conversations"
            endpoint = f"{prefix}/{chat['conversation_id']}/messages"
            response = await owner.post(
                endpoint,
                json={
                    "text": text,
                    "client_message_id": str(uuid4()),
                    "revision": chat["revision"],
                },
                headers={
                    "X-Conversation-Token": chat["conversation_token"],
                    "Idempotency-Key": uuid4().hex,
                },
            )
            assert response.status_code == 200, response.text
            return response.json()

        target_version = await train(target_id, target, 1)
        handoff["config"].update(
            target_agent_id=str(target_id), target_version_id=str(target_version)
        )
        short_version = await train(root_id, root, 1, minimum=False)
        rejected = await owner.get(f"{root_url}/versions/{short_version}/release-check")
        assert "release_test_minimum" in {item["code"] for item in rejected.json()["errors"]}

        first_version = await train(root_id, root, 2)
        check_url = f"{root_url}/versions/{first_version}/release-check"
        checked = (await owner.get(check_url)).json()
        assert [item["code"] for item in checked["errors"]] == ["release_smoke_required"]
        first_chat = await start(first_version)
        assert first_chat["messages"] == ["Где заказ 321?"]
        assert (await say(first_chat, "Завтра"))["status"] == "completed"
        old_chat = await start(first_version)
        etag = (await owner.get(root_url)).headers["etag"]
        publish_url = f"{root_url}/releases"
        request = {"version_id": str(first_version)}
        assert (
            await editor.post(publish_url, json=request, headers={"If-Match": etag})
        ).status_code == 403
        assert (await owner.post(publish_url, json=request)).status_code == 428
        warning = await owner.post(publish_url, json=request, headers={"If-Match": etag})
        assert warning.status_code == 409
        assert warning.json()["error"]["code"] == "release_quality_confirmation_required"
        request.update(warning_acknowledgement=True, reason="Проверено вручную")
        first, concurrent = await asyncio.gather(
            owner.post(publish_url, json=request, headers={"If-Match": etag}),
            owner.post(publish_url, json=request, headers={"If-Match": etag}),
        )
        assert sorted([first.status_code, concurrent.status_code]) == [201, 412]
        release_a = (first if first.status_code == 201 else concurrent).json()
        assert {pin["version_id"] for pin in release_a["dependency_manifest"]} == {
            str(first_version),
            str(target_version),
        }
        assert (await owner.post(f"{target_url}/archive")).status_code == 409

        # Publish a root version that no longer calls the target.
        root["flow"]["edges"][0]["target"] = root["flow"]["nodes"][3]["id"]
        root["flow"]["nodes"] = [
            node for node in root["flow"]["nodes"] if node["id"] != handoff["id"]
        ]
        root["flow"]["edges"] = [
            edge for edge in root["flow"]["edges"] if edge["source"] != handoff["id"]
        ]
        second_version = await train(root_id, root, 3)
        assert (await start(second_version))["status"] == "completed"
        etag = (await owner.get(root_url)).headers["etag"]
        second = await owner.post(
            publish_url,
            json={**request, "version_id": str(second_version)},
            headers={"If-Match": etag},
        )
        assert second.status_code == 201, second.text
        assert (await owner.post(f"{target_url}/archive")).status_code == 200
        assert (await say(old_chat, "Сегодня"))["messages"] == ["Доставка: Сегодня / важно"]
        undo = {"release_id": release_a["id"], "reason": "Возвращаем проверенный сценарий"}
        assert (
            await owner.post(
                f"{root_url}/rollback", json=undo, headers={"If-Match": second.headers["etag"]}
            )
        ).status_code == 409
        assert (await owner.post(f"{target_url}/restore")).status_code == 200
        rolled = await owner.post(
            f"{root_url}/rollback", json=undo, headers={"If-Match": second.headers["etag"]}
        )
        assert rolled.status_code == 200 and rolled.json()["current_release_id"] == release_a["id"]
        assert (await owner.get(root_url)).json()["current_release_id"] == release_a["id"]
        history = (await owner.get(publish_url)).json()["items"]
        assert history == [second.json(), release_a]
        assert [
            agent["id"] for agent in (await owner.get(f"{base}?published=true")).json()["items"]
        ] == [str(root_id)]
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(Release, UUID(release_a["id"])) is None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            audit = (
                await db.scalars(
                    select(AuditEvent).where(
                        AuditEvent.action.in_(("agent.published", "agent.rolled_back"))
                    )
                )
            ).all()
            assert len(audit) == 3 and any(
                event.details["reason"] == undo["reason"] for event in audit
            )
            published = await db.get(Release, UUID(release_a["id"]))
            assert published is not None
            with pytest.raises(DBAPIError):
                async with db.begin_nested():
                    published.reason = "Попытка изменить публикацию"
                    await db.flush()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.scalar(select(Release.id)) is None
            assert await db.scalar(select(ReleaseVersion.version_id)) is None
