import os
from uuid import UUID, uuid4

import pytest

from benchmarks.nlu.demo import ANSWERS, SETTINGS, demo_config, load_demo_fixtures
from dialogentis.datasets.export_worker import serialize_export
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


@pytest.mark.parametrize("name", ["order_support", "education"])
async def test_demo_agent_imports_trains_and_answers(app, name):
    fixture = load_demo_fixtures()[name]
    config = demo_config(name, fixture)
    factory = app.state.session_factory

    async with client_for(app) as owner:
        _, org_id, agent_id = await create_agent(app, owner)
        organization = f"/api/v1/organizations/{org_id}"
        agent = f"{organization}/agents/{agent_id}"
        uploaded = await owner.post(
            f"{agent}/imports/preview",
            files={"file": (f"{name}.json", serialize_export(config, "json"), "application/json")},
            data={"mode": "replace"},
        )
        assert uploaded.status_code == 202, uploaded.text
        preview_id = UUID(uploaded.json()["preview_id"])
        await mark_published(app, org_id)
        await process_job(factory, preview_id, org_id)
        preview = await owner.get(f"{agent}/imports/{preview_id}")
        assert preview.status_code == 200 and preview.json()["can_apply"], preview.text

        grant = await owner.post(f"{agent}/edit-lease", json={"tab_id": str(uuid4())})
        assert grant.status_code == 200, grant.text
        lease = grant.json()
        draft = await owner.post(
            f"{agent}/imports/{preview_id}/apply",
            headers={
                "X-Edit-Lease": lease["lease_token"],
                "X-Edit-Fence": str(lease["fencing_counter"]),
                "If-Match": '"rev-1"',
            },
        )
        assert draft.status_code == 200, draft.text
        assert draft.json()["config"]["nlu_settings"] == SETTINGS[name].model_dump()

        started = await owner.post(
            f"{agent}/versions",
            json={"source_revision": draft.json()["revision"]},
            headers={"Idempotency-Key": uuid4().hex},
        )
        assert started.status_code == 202, started.text
        version_id, job_id = (UUID(started.json()[key]) for key in ("version_id", "job_id"))
        await mark_published(app, org_id)
        await process_job(factory, job_id, org_id, HANDLERS, "training")
        version = await owner.get(f"{agent}/versions/{version_id}")
        assert version.status_code == 200 and version.json()["status"] == "ready", version.text
        report = await owner.get(f"{agent}/versions/{version_id}/report")
        assert report.status_code == 200, report.text
        quality = report.json()
        assert quality["training_examples"] == sum(
            len(intent["train"]) for intent in fixture["intents"]
        )
        assert quality["intents"]["known_count"] == 25
        assert quality["intents"]["unknown_count"] == 20
        assert quality["intents"]["macro_f1"] >= 0.85
        assert quality["intents"]["unknown_rejection"] >= 0.80
        assert quality["entities"]["f1"] >= 0.85

        opened = await owner.post(
            f"{agent}/test-conversations", json={"version_id": str(version_id)}
        )
        assert opened.status_code == 201, opened.text
        chat = opened.json()
        intent = fixture["intents"][0]
        response = await owner.post(
            f"{organization}/test-conversations/{chat['conversation_id']}/messages",
            headers={
                "X-Conversation-Token": chat["conversation_token"],
                "Idempotency-Key": uuid4().hex,
            },
            json={
                "client_message_id": str(uuid4()),
                "revision": chat["revision"],
                "text": intent["train"][0],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["messages"] == [ANSWERS[intent["code"]]]

        edited = draft.json()["config"]
        message = next(node for node in edited["flow"]["nodes"] if node["type"] == "say")
        original = message["config"]["text"]
        message["config"]["text"] = "Ответ более свежего черновика"
        saved = await owner.put(
            f"{agent}/draft",
            headers={
                "X-Edit-Lease": lease["lease_token"],
                "X-Edit-Fence": str(lease["fencing_counter"]),
                "If-Match": draft.headers["etag"],
            },
            json={"schema_version": 1, "config": edited},
        )
        assert saved.status_code == 200, saved.text
        body = {"format": "json", "source": "version", "version_id": str(version_id)}
        unrelated = await owner.post(
            f"{organization}/agents", json={"slug": "unrelated", "name": "Другой агент"}
        )
        assert unrelated.status_code == 201
        assert (
            await owner.post(f"{organization}/agents/{unrelated.json()['id']}/exports", json=body)
        ).status_code == 404
        assert (
            await owner.post(f"{agent}/exports", json={**body, "source": "draft"})
        ).status_code == 422
        export = await owner.post(f"{agent}/exports", json=body)
        assert export.status_code == 202, export.text
        assert export.json()["revision"] == draft.json()["revision"]
        export_id = UUID(export.json()["export_id"])
        await mark_published(app, org_id)
        await process_job(factory, export_id, org_id)
        downloaded = await owner.get(f"{organization}/exports/{export_id}/download")
        assert downloaded.status_code == 200
        flow = downloaded.json()["config"]["flow"]
        exported = next(node for node in flow["nodes"] if node["id"] == message["id"])
        assert exported["config"]["text"] == original
