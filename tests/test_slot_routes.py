import os
from uuid import uuid4

import pytest

from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_slot_save_copy_validation_and_references(app):
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
        slot_id = str(uuid4())
        slot = {
            "id": slot_id,
            "code": "amount",
            "name": "Сумма",
            "value_type": "number",
            "default": "100.25",
            "required": True,
            "sensitive": True,
        }
        config = draft["config"]
        config["slots"] = [slot]
        config["flow"]["nodes"][2]["config"]["text"] = "Сумма: {{amount}}"
        body = {"schema_version": 1, "config": config}
        saved = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert saved.status_code == 200 and saved.json()["config"]["slots"][0] == slot
        copy = await owner.post(f"{base}/copy", json={"name": "Copy", "slug": "copy"})
        assert copy.status_code == 201
        cloned = (
            await owner.get(f"/api/v1/organizations/{org_id}/agents/{copy.json()['id']}/draft")
        ).json()["config"]
        assert cloned["slots"][0]["id"] != slot_id and cloned["slots"][0]["default"] == "100.25"
        headers["If-Match"] = '"rev-2"'
        slot["default"] = 100.25
        invalid = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert invalid.status_code == 422
        slot["default"] = "100.25"
        config["slots"] = []
        rejected = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert rejected.status_code == 422
        assert "slot_in_use" in {issue["code"] for issue in rejected.json()["error"]["details"]}
        config["flow"]["nodes"][2]["config"]["text"] = "Укажите сумму"
        assert (await owner.put(f"{base}/draft", headers=headers, json=body)).status_code == 200


async def test_direct_api_cannot_remove_slot_or_entity_used_by_ask_block(app):
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
        slot_id, entity_id = str(uuid4()), str(uuid4())
        config = draft["config"]
        config["slots"] = [{"id": slot_id, "code": "city"}]
        config["entities"] = [
            {
                "id": entity_id,
                "code": "city",
                "name": "Город",
                "value_type": "string",
                "method": "dictionary",
                "dictionary": [{"value": "Омск"}],
            }
        ]
        config["flow"]["nodes"].append(
            {
                "id": str(uuid4()),
                "type": "ask_slot",
                "config": {"slot_id": slot_id, "entity_id": entity_id, "prompt": "Город?"},
                "position": {"x": 0, "y": 0},
            }
        )
        body = {"schema_version": 1, "config": config}
        assert (await owner.put(f"{base}/draft", headers=headers, json=body)).status_code == 200
        headers["If-Match"] = '"rev-2"'
        config["slots"] = []
        config["entities"] = []
        response = await owner.put(f"{base}/draft", headers=headers, json=body)
        assert response.status_code == 422
        assert {"unknown_ask_slot", "unknown_ask_entity"} <= {
            item["code"] for item in response.json()["error"]["details"]
        }
