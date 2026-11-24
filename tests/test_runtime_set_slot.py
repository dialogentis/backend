from copy import deepcopy
from uuid import uuid4

import pytest

from dialogentis.agents import copy_config, empty_config
from dialogentis.datasets.json_transfer import PortableConfig
from dialogentis.flow_schema import Flow
from dialogentis.flow_validation import validate_flow
from dialogentis.runtime.core import RuntimeEngine
from dialogentis.set_slot import validate_set_references
from dialogentis.slots import validate_slots
from tests.dialogue_fixtures import clarify
from tests.test_runtime_core import edge, node


def configuration(operation="set", source="constant", value=0):
    config = empty_config()
    target, original = uuid4(), uuid4()
    config["slots"] = [
        {"id": str(target), "code": "target", "value_type": "integer", "sensitive": True},
        {"id": str(original), "code": "original", "value_type": "integer"},
    ]
    start, switch, _, end = config["flow"]["nodes"]
    assign = node(
        "set_slot",
        slot_id=str(target),
        operation=operation,
        source=source,
        source_slot_id=str(original) if source == "slot" else None,
        value=value,
    )
    response = node("say", text="{{target}}")
    config["flow"]["nodes"] += [assign, response]
    config["flow"]["edges"][0]["target"] = assign["id"]
    config["flow"]["edges"] += [
        edge(assign, response),
        edge(response, switch),
        edge(assign, end, "error"),
    ]
    return config, assign, target, original


def run(config, values=None):
    version = uuid4()
    return RuntimeEngine(version, config, clarify).start(values)


def test_assignment_constant_copy_clear_and_missing_source():
    config, _, _, _ = configuration()
    result = run(config)
    assert dict(result.state.slot_values)["target"] == 0 and result.messages == ("0",)
    config, _, _, _ = configuration(source="slot", value=None)
    result = run(config, {"original": 23})
    assert dict(result.state.slot_values)["target"] == 23 and result.messages == ("23",)
    assert run(config).state.status == "completed"
    config, _, _, _ = configuration(operation="clear", value=None)
    result = run(config, {"target": 10})
    assert dict(result.state.slot_values)["target"] is None and result.messages == ("",)


def test_invalid_assignment_and_outputs_rejected():
    config, assignment, _, _ = configuration(value=True)
    errors, _ = validate_flow(config)
    assert "invalid_set_slot" in {issue.code for issue in errors}
    assignment["config"]["value"] = 1
    config["flow"]["edges"] = [
        item for item in config["flow"]["edges"] if item["source_handle"] != "error"
    ]
    assert "missing_output" in {issue.code for issue in validate_flow(config)[0]}
    assignment["config"].update(source="slot", source_slot_id=str(uuid4()))
    assert "invalid_set_slot" in {issue.code for issue in validate_flow(config)[0]}


def test_copy_and_portable_config_preserve_and_remap_references():
    config, _, target, original = configuration(source="slot", value=None)
    PortableConfig.model_validate(config)
    clone = copy_config(config)
    assign = next(n for n in clone["flow"]["nodes"] if n["type"] == "set_slot")
    slots = {item["code"]: item["id"] for item in clone["slots"]}
    assert assign["config"]["slot_id"] == slots["target"] != str(target)
    assert assign["config"]["source_slot_id"] == slots["original"] != str(original)
    invalid = deepcopy(config)
    next(n for n in invalid["flow"]["nodes"] if n["type"] == "set_slot")["config"][
        "source_slot_id"
    ] = str(uuid4())
    with pytest.raises(ValueError):
        PortableConfig.model_validate(invalid)


def test_references_protect_both_slots_and_source_types():
    config, assignment, _, _ = configuration(source="slot", value=None)
    slots, _ = validate_slots(config["slots"])
    flow = Flow.model_validate(config["flow"])
    assert not validate_set_references(flow, slots)
    assert validate_set_references(flow, slots[:1])[0].code == "unknown_set_slot"
    config["slots"][1]["value_type"] = "boolean"
    assert "invalid_set_slot" in {issue.code for issue in validate_flow(config)[0]}
    assignment["config"]["operation"] = "clear"
    assert "invalid_set_slot" not in {issue.code for issue in validate_flow(config)[0]}
