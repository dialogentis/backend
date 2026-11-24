from uuid import uuid4

import pytest

from dialogentis.conditions import ConditionConfig, choose_branch, validate_condition
from dialogentis.slots import Slot


def test_ordered_branches_decimal_and_zero():
    slot = Slot(id=uuid4(), code="amount", value_type="number")
    branches = [uuid4(), uuid4()]
    config = ConditionConfig.model_validate(
        {
            "branches": [
                {
                    "id": str(branches[0]),
                    "expression": {
                        "op": "gte",
                        "slot_id": str(slot.id),
                        "value": "9007199254740993.01",
                    },
                },
                {"id": str(branches[1]), "expression": {"op": "exists", "slot_id": str(slot.id)}},
            ]
        }
    )
    slots = {slot.id: slot}
    validate_condition(config, slots)
    assert choose_branch(config, slots, {"amount": "9007199254740993.02"}) == str(branches[0])
    assert choose_branch(config, slots, {"amount": "0"}) == str(branches[1])
    assert choose_branch(config, slots, {"amount": None}) == "else"


def test_false_exists_nested_and_type_validation():
    slot = Slot(id=uuid4(), code="flag", value_type="boolean")
    slots = {slot.id: slot}
    leaf = {"op": "eq", "slot_id": str(slot.id), "value": False}
    branch = uuid4()
    config = ConditionConfig.model_validate(
        {
            "branches": [
                {
                    "id": str(branch),
                    "expression": {
                        "op": "all",
                        "conditions": [leaf, {"op": "exists", "slot_id": str(slot.id)}],
                    },
                }
            ]
        }
    )
    validate_condition(config, slots)
    assert choose_branch(config, slots, {"flag": False}) == str(branch)
    assert choose_branch(config, slots, {}) == "else"
    leaf["value"] = 0
    bad = ConditionConfig.model_validate({"branches": [{"id": str(branch), "expression": leaf}]})
    with pytest.raises(ValueError):
        validate_condition(bad, slots)
    deep = {
        "op": "all",
        "conditions": [{"op": "any", "conditions": [{"op": "all", "conditions": [leaf]}]}],
    }
    with pytest.raises(ValueError):
        validate_condition(
            ConditionConfig.model_validate({"branches": [{"id": str(branch), "expression": deep}]}),
            slots,
        )


def test_runtime_and_graph_use_branch_handles():
    from dialogentis.agents import empty_config
    from dialogentis.flow_validation import validate_flow
    from dialogentis.runtime.core import RuntimeEngine
    from tests.dialogue_fixtures import clarify
    from tests.test_runtime_core import edge, node

    config = empty_config()
    slot, branch, version = uuid4(), uuid4(), uuid4()
    config["slots"] = [{"id": str(slot), "code": "amount", "value_type": "integer"}]
    start, switch, fallback, end = config["flow"]["nodes"]
    conditional = node(
        "condition",
        branches=[
            {"id": str(branch), "expression": {"op": "gte", "slot_id": str(slot), "value": 10}}
        ],
    )
    config["flow"]["nodes"].append(conditional)
    config["flow"]["edges"][0]["target"] = conditional["id"]
    config["flow"]["edges"] += [
        edge(conditional, switch, str(branch)),
        edge(conditional, end, "else"),
    ]
    assert not validate_flow(config)[0]
    engine = RuntimeEngine(version, config, clarify)
    assert engine.start({"amount": 10}).state.status == "waiting_input"
    assert engine.start({"amount": 9}).state.status == "completed"
    config["flow"]["edges"].pop()
    assert "missing_output" in {item.code for item in validate_flow(config)[0]}


def test_nested_conditions_copy_import_and_draft_references():
    from copy import deepcopy

    from fastapi import HTTPException

    from dialogentis.agents import copy_config, empty_config
    from dialogentis.datasets.json_transfer import PortableConfig
    from dialogentis.datasets.routes import validate_configuration
    from tests.test_runtime_core import edge, node

    original = empty_config()
    slot, branch = uuid4(), uuid4()
    original["slots"] = [{"id": str(slot), "code": "amount", "value_type": "integer"}]
    start, switch, _, end = original["flow"]["nodes"]
    expression = {
        "op": "all",
        "conditions": [
            {"op": "exists", "slot_id": str(slot)},
            {"op": "gte", "slot_id": str(slot), "value": 0},
        ],
    }
    condition = node("condition", branches=[{"id": str(branch), "expression": expression}])
    original["flow"]["nodes"].append(condition)
    original["flow"]["edges"][0]["target"] = condition["id"]
    original["flow"]["edges"] += [
        edge(condition, switch, str(branch)),
        edge(condition, end, "else"),
    ]
    PortableConfig.model_validate(original)
    clone = copy_config(original)
    cloned = next(item for item in clone["flow"]["nodes"] if item["type"] == "condition")
    new_slot = clone["slots"][0]["id"]
    new_branch = cloned["config"]["branches"][0]
    assert new_slot != str(slot) and new_branch["id"] != str(branch)
    assert all(child["slot_id"] == new_slot for child in new_branch["expression"]["conditions"])
    assert any(
        item["source"] == cloned["id"] and item["source_handle"] == new_branch["id"]
        for item in clone["flow"]["edges"]
    )
    PortableConfig.model_validate(clone)
    bad = deepcopy(original)
    bad["slots"] = []
    assert "unknown_condition_slot" in {issue.code for issue in validate_configuration(bad)[0]}
    with pytest.raises(ValueError):
        PortableConfig.model_validate(bad)
    with pytest.raises(HTTPException):
        copy_config(bad)


def test_incomplete_conditions_can_be_copied_and_transferred():
    from dialogentis.agents import copy_config, empty_config
    from dialogentis.datasets.json_transfer import PortableConfig
    from tests.test_runtime_core import node

    draft = empty_config()
    branch = uuid4()
    conditional = node(
        "condition",
        branches=[{"id": str(branch), "expression": {"op": "eq", "slot_id": None, "value": ""}}],
    )
    draft["flow"]["nodes"].append(conditional)
    PortableConfig.model_validate(draft)
    clone = copy_config(draft)
    item = next(item for item in clone["flow"]["nodes"] if item["type"] == "condition")
    assert item["config"]["branches"][0]["id"] != str(branch)
    assert item["config"]["branches"][0]["expression"]["slot_id"] is None
    PortableConfig.model_validate(clone)
