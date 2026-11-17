from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi import HTTPException

from dialogentis.agents import copy_config, empty_config
from dialogentis.datasets.routes import validate_configuration
from dialogentis.flow_validation import validate_flow
from dialogentis.http_action import HTTP_OUTPUTS


def configuration():
    config = empty_config()
    source, target, integration = (str(uuid4()) for _ in range(3))
    config["slots"] = [{"id": source, "code": "order"}, {"id": target, "code": "status"}]
    action = {
        "id": str(uuid4()),
        "type": "http_action",
        "position": {"x": 0, "y": 0},
        "config": {
            "integration_id": integration,
            "method": "POST",
            "path": "/orders",
            "query": {"order": source},
            "body": {"order_id": source},
            "response_mapping": {"status": target},
            "mock_response": {"status": "готов"},
        },
    }
    config["flow"]["nodes"].append(action)
    end = config["flow"]["nodes"][3]
    config["flow"]["edges"] += [
        {"id": str(uuid4()), "source": action["id"], "target": end["id"], "source_handle": output}
        for output in HTTP_OUTPUTS
    ]
    return config, action


def test_http_action_validation_rejects_unsafe_inputs_and_missing_outputs():
    config, action = configuration()
    assert not validate_flow(config)[0]
    for path in ("https://example.com/x", "//example.com/x", "/../x", "/%2e%2e/x", "/x?q=1"):
        action["config"]["path"] = path
        assert "invalid_http_action" in {i.code for i in validate_flow(config)[0]}
    action["config"]["path"] = "/orders"
    action["config"]["body"]["order_id"] = str(uuid4())
    assert "invalid_http_action" in {i.code for i in validate_flow(config)[0]}
    assert "invalid_http_action" in {i.code for i in validate_configuration(config)[0]}
    action["config"]["body"] = {}
    action["config"]["response_mapping"]["duplicate"] = action["config"]["response_mapping"][
        "status"
    ]
    assert "invalid_http_action" in {i.code for i in validate_flow(config)[0]}
    del action["config"]["response_mapping"]["duplicate"]
    config["flow"]["edges"].pop()
    assert "missing_output" in {i.code for i in validate_flow(config)[0]}
    config["flow"]["edges"][-1]["source_handle"] = "other"
    assert "invalid_output" in {i.code for i in validate_flow(config)[0]}


def test_http_action_copy_remaps_slots_but_keeps_integration_and_mock():
    config, action = configuration()
    original = deepcopy(config)
    clone = copy_config(config)
    copied = next(node for node in clone["flow"]["nodes"] if node["type"] == "http_action")
    assert config == original
    assert copied["config"]["integration_id"] == action["config"]["integration_id"]
    assert copied["config"]["query"]["order"] == clone["slots"][0]["id"]
    assert copied["config"]["body"]["order_id"] == clone["slots"][0]["id"]
    assert copied["config"]["response_mapping"]["status"] == clone["slots"][1]["id"]
    assert copied["config"]["mock_response"] == {"status": "готов"}
    action["config"]["query"]["bad"] = str(uuid4())
    with pytest.raises(HTTPException):
        copy_config(config)
