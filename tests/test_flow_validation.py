from copy import deepcopy
from uuid import uuid4

from dialogentis.agents import empty_config
from dialogentis.flow_validation import validate_flow


def codes(issues):
    return {item.code for item in issues}


def node(kind):
    return {"id": str(uuid4()), "type": kind, "config": {}, "position": {"x": 0, "y": 0}}


def edge(source, target, handle="next"):
    return {
        "id": str(uuid4()),
        "source": source["id"],
        "target": target["id"],
        "source_handle": handle,
    }


def test_default_flow_and_partial_validation():
    errors, warnings = validate_flow(empty_config())
    assert not errors
    assert codes(warnings) == {"unreachable_node", "missing_fallback_example"}


def test_fallback_example_must_be_explicit_plain_text():
    config = empty_config()
    fallback = config["flow"]["nodes"][2]
    for text in ("<b>Текст</b>", "{{city}}", "x" * 501):
        fallback["config"]["example"] = text
        assert "invalid_fallback_example" in codes(validate_flow(config)[0])
    fallback["config"]["example"] = "Где мой заказ?"
    errors, warnings = validate_flow(config)
    assert not errors and "missing_fallback_example" not in codes(warnings)


def test_missing_connections_duplicate_ids_and_start_incoming():
    config = empty_config()
    flow = config["flow"]
    start, switch, fallback, end = flow["nodes"]
    flow["edges"] = [
        edge(start, switch),
        edge(start, end),
        edge(end, start),
        edge(switch, {"id": str(uuid4())}, "fallback"),
    ]
    duplicate = deepcopy(end)
    flow["nodes"].append(duplicate)
    flow["edges"].append(deepcopy(flow["edges"][0]))
    fallback["config"]["resume_node_id"] = str(uuid4())
    errors, warnings = validate_flow(config)
    assert {
        "duplicate_node",
        "duplicate_edge",
        "duplicate_handle",
        "start_has_input",
        "invalid_output",
        "missing_target",
        "invalid_resume",
    } <= codes(errors)
    assert "unreachable_node" not in codes(warnings)  # Duplicate IDs make reachability ambiguous.


def test_missing_start_required_outputs_and_invalid_flow_are_reported_without_input():
    config = empty_config()
    config["flow"]["nodes"] = [node("say"), node("switch_intent")]
    config["flow"]["edges"] = []
    errors, _ = validate_flow(config)
    assert "start_count" in codes(errors)
    assert sum(item.code == "missing_output" for item in errors) == 2
    errors, _ = validate_flow({"flow": {"PRIVATE_VALUE": "SECRET_SENTINEL"}})
    assert codes(errors) == {"invalid_flow"} and "SECRET_SENTINEL" not in repr(errors)


def test_unreachable_nodes_and_resume_links():
    config = empty_config()
    flow = config["flow"]
    fallback, end = flow["nodes"][2:]
    fallback["config"]["resume_node_id"] = end["id"]
    fallback["config"]["example"] = "Где мой заказ?"
    errors, warnings = validate_flow(config)
    assert not errors and not warnings
    fallback["config"]["resume_node_id"] = str(uuid4())
    errors, warnings = validate_flow(config)
    assert "invalid_resume" in codes(errors)
    assert any(item.path == ("flow", "nodes", 3) for item in warnings)
    fallback["config"]["resume_node_id"] = flow["nodes"][0]["id"]
    errors, _ = validate_flow(config)
    assert any(
        item.code == "start_has_input" and item.path[-1] == "resume_node_id" for item in errors
    )


def test_immediate_cycles_and_waiting_paths():
    config = empty_config()
    start, switch, fallback, _ = config["flow"]["nodes"]
    response = node("say")
    ask = node("ask_slot")
    config["flow"]["nodes"] += [response, ask]
    config["flow"]["edges"] += [
        edge(switch, response, "intent"),
        edge(response, ask),
        edge(ask, response, "success"),
    ]
    errors, _ = validate_flow(config)
    assert "immediate_cycle" in codes(errors)
    # Returning from fallback always waits for a new message, so that loop is safe.
    config["flow"]["edges"] = [edge(start, switch), edge(switch, fallback, "fallback")]
    config["flow"]["nodes"] = [start, switch, fallback]
    assert "immediate_cycle" not in codes(validate_flow(config)[0])
    # An invalid/exhausted branch of ask_slot follows user input, unlike success.
    config["flow"]["nodes"] += [response, ask]
    config["flow"]["edges"] += [edge(response, ask), edge(ask, response, "invalid")]
    assert "immediate_cycle" not in codes(validate_flow(config)[0])


def test_switch_intent_outputs_check_the_referenced_intent():
    config = empty_config()
    switch, end = config["flow"]["nodes"][1], config["flow"]["nodes"][3]
    config["flow"]["nodes"][2]["config"]["example"] = "Где мой заказ?"
    intent_id = str(uuid4())
    config["intents"] = [{"id": intent_id, "code": "order", "name": "Заказ", "enabled": False}]
    config["flow"]["edges"].append(edge(switch, end, intent_id))
    errors, warnings = validate_flow(config)
    assert not errors and "disabled_intent_output" in codes(warnings)
    config["intents"][0]["enabled"] = True
    assert not validate_flow(config)[1]
    config["intents"].clear()
    errors, _ = validate_flow(config)
    assert any(
        item.code == "unknown_intent_output" and item.path[-1] == "source_handle" for item in errors
    )


def test_ask_slot_configuration_and_all_three_outputs():
    config = empty_config()
    slot_id, entity_id = str(uuid4()), str(uuid4())
    config["slots"] = [{"id": slot_id, "code": "city", "value_type": "string"}]
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
    ask = node("ask_slot")
    ask["config"] = {
        "slot_id": slot_id,
        "entity_id": entity_id,
        "prompt": "Укажите город",
        "max_attempts": 2,
        "checks": [{"kind": "min_length", "value": 2}],
    }
    config["flow"]["nodes"].append(ask)
    end = config["flow"]["nodes"][3]
    config["flow"]["edges"] += [
        edge(ask, end, handle) for handle in ("success", "invalid", "exhausted")
    ]
    assert not validate_flow(config)[0]
    config["flow"]["edges"].pop()
    assert "missing_output" in codes(validate_flow(config)[0])
    config["entities"][0]["value_type"] = "integer"
    assert "entity_type_mismatch" in codes(validate_flow(config)[0])
    ask["config"]["checks"] = [{"kind": "pattern", "pattern": r"(a)\1"}]
    assert "invalid_ask_slot" in codes(validate_flow(config)[0])


def test_structural_check_handles_limits():
    config = empty_config()
    nodes = [node("start"), *[node("condition") for _ in range(299)]]
    edges = [edge(nodes[i], nodes[i + 1], "next" if i == 0 else "branch-a") for i in range(299)]
    edges += [edge(nodes[i % 298 + 1], nodes[i % 298 + 2], f"extra-{i}") for i in range(301)]
    config["flow"]["nodes"], config["flow"]["edges"] = nodes, edges
    errors, warnings = validate_flow(config)
    assert len(nodes) == 300 and len(edges) == 600
    assert not errors and not warnings
