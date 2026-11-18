from uuid import UUID, uuid4

from dialogentis.agents import copy_config, empty_config
from dialogentis.datasets.export_worker import serialize_export
from dialogentis.datasets.json_transfer import parse_json
from dialogentis.runtime.core import Prediction, RuntimeEngine, RuntimeState
from tests.test_runtime_core import edge, node


def call_flows():
    root, target = empty_config(), empty_config()
    root_version, target_version, target_agent = uuid4(), uuid4(), uuid4()
    sent, result, preserved, incoming, response = [str(uuid4()) for _ in range(5)]
    root["slots"] = [
        {"id": sent, "code": "order_number", "default": "321"},
        {"id": result, "code": "delivery", "value_type": "string"},
        {"id": preserved, "code": "preserved", "default": "важно"},
    ]
    target["slots"] = [
        {"id": incoming, "code": "ticket"},
        {"id": response, "code": "answer"},
        {"id": str(uuid4()), "code": "preserved", "default": "чужое"},
    ]
    call = node(
        "handoff",
        target_version_id=str(target_version),
        target_agent_id=str(target_agent),
        input_mapping={sent: incoming},
        return_mapping={response: result},
    )
    say = node("say", text="Доставка: {{delivery}} / {{preserved}}")
    root_end = root["flow"]["nodes"][3]
    root["flow"]["nodes"] += [call, say]
    root["flow"]["edges"][0]["target"] = call["id"]
    root["flow"]["edges"] += [
        edge(call, say, "returned"),
        edge(say, root_end),
        edge(call, root_end, "error"),
    ]
    ask = node("ask_slot", slot_id=response, prompt="Где заказ {{ticket}}?")
    returned = node("return", slot_ids=[response])
    target_end = target["flow"]["nodes"][3]
    target["flow"]["nodes"] += [ask, returned]
    target["flow"]["edges"][0]["target"] = ask["id"]
    target["flow"]["edges"] += [
        edge(ask, returned, "success"),
        edge(ask, ask, "invalid"),
        edge(ask, target_end, "exhausted"),
    ]
    return root, target, root_version, target_version, call, returned


def engines(root, target, root_version, target_version):
    versions = {}
    for version, config in [(root_version, root), (target_version, target)]:
        versions[version] = RuntimeEngine(
            version,
            config,
            lambda _, version=version: Prediction(version, None, ("low_score",)),
            resolve_version=versions.get,
        )
    return versions[root_version]


def test_call_persists_frames_and_returns_only_explicit_parameters():
    root, target, root_version, target_version, call, _ = call_flows()
    start = engines(root, target, root_version, target_version).start()
    assert start.messages == ("Где заказ 321?",)
    assert start.state.version_id == root_version
    assert start.state.active_path == (target_version,)
    assert start.state.callers[0].node_id == UUID(call["id"])
    checkpoint = RuntimeState.model_validate_json(start.state.model_dump_json())
    result = engines(root, target, root_version, target_version).send(checkpoint, "Завтра")
    assert result.state.status == "completed"
    assert result.messages == ("Доставка: Завтра / важно",)
    assert not result.state.callers and not result.state.active_path
    assert "Завтра" not in repr(result.trace)
    assert any(event.event == "returned" for event in result.trace)


def test_transfer_drops_caller_and_root_return_completes_the_session():
    root, target, root_version, target_version, call, _ = call_flows()
    call["config"].update(mode="transfer", return_mapping={})
    root["flow"]["edges"] = [
        item for item in root["flow"]["edges"] if item["source_handle"] != "returned"
    ]
    engine = engines(root, target, root_version, target_version)
    start = engine.start()
    assert start.messages == ("Где заказ 321?",)
    assert start.state.active_path == (target_version,) and not start.state.callers
    result = engine.send(start.state, "Завтра")
    assert result.state.status == "completed" and result.messages == ()


def test_recursive_call_fails_safely_and_sensitive_values_cannot_be_downgraded():
    root, target, root_version, target_version, call, _ = call_flows()
    root["slots"][0]["sensitive"] = True
    denied = engines(root, target, root_version, target_version).start()
    assert denied.state.status == "completed" and denied.messages == ()
    assert any(event.output == "error" for event in denied.trace)
    call["config"]["target_version_id"] = str(root_version)
    failed = engines(root, target, root_version, target_version).start()
    assert failed.state.status == "failed" and failed.error_code == "handoff_cycle"


def test_handoff_copy_only_remaps_local_parameters_and_json_keeps_pinned_version():
    root, target, _, target_version, call, returned = call_flows()
    clone = copy_config(root)
    config = next(node["config"] for node in clone["flow"]["nodes"] if node["type"] == "handoff")
    assert config["target_version_id"] == str(target_version)
    assert config["input_mapping"] == {
        clone["slots"][0]["id"]: next(iter(call["config"]["input_mapping"].values()))
    }
    assert config["return_mapping"] == {
        next(iter(call["config"]["return_mapping"])): clone["slots"][1]["id"]
    }
    loaded, errors = parse_json(serialize_export(target, "json"))
    assert not errors and loaded is not None
    assert (
        next(node.config for node in loaded.flow.nodes if node.type == "return")
        == returned["config"]
    )


def test_only_explicitly_forwarded_current_text_reaches_next_agent():
    root, target, root_version, target_version, call, _ = call_flows()
    start, switch, _, _ = root["flow"]["nodes"][:4]
    root["flow"]["edges"][0] = edge(start, switch)
    root["flow"]["edges"] = [
        item for item in root["flow"]["edges"] if item["source"] != switch["id"]
    ]
    root["flow"]["edges"].append(edge(switch, call, "fallback"))
    target_start, target_switch = target["flow"]["nodes"][:2]
    target["flow"]["edges"][0] = edge(target_start, target_switch)
    texts = []
    second = RuntimeEngine(
        target_version,
        target,
        lambda text: texts.append(text) or Prediction(target_version, None, ("low_score",)),
    )
    for forward in (False, True):
        call["config"]["forward_current_message"] = forward
        first = RuntimeEngine(
            root_version,
            root,
            lambda _: Prediction(root_version, None, ("low_score",)),
            resolve_version={target_version: second}.get,
        )
        episode = first.start()
        result = first.send(episode.state, "Где заказ?")
        assert result.state.active_path == (target_version,)
        assert texts == (["Где заказ?"] if forward else [])
