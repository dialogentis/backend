from copy import deepcopy
from uuid import UUID, uuid4

from dialogentis.http_action import MockIntegration
from dialogentis.runtime.core import Prediction, RuntimeEngine
from tests.test_http_action_config import configuration
from tests.test_runtime_core import edge, node


def test_mock_action_maps_only_allowed_fields_and_branches_without_leaking_values():
    config, action = configuration()
    config["slots"][0]["sensitive"] = True
    config["slots"][1]["sensitive"] = True
    _, switch, _, end = config["flow"]["nodes"][:4]
    ask = node(
        "ask_slot",
        slot_id=config["slots"][0]["id"],
        prompt="Укажите номер заказа",
        checks=[{"kind": "pattern", "pattern": "[0-9]{5}"}],
    )
    retry = node("say", text="Нужны пять цифр")
    reply = node("say", text="Статус: {{status}}")
    config["flow"]["edges"][0]["target"] = ask["id"]
    config["flow"]["nodes"] += [ask, retry, reply]
    next(
        e
        for e in config["flow"]["edges"]
        if e["source"] == action["id"] and e["source_handle"] == "success"
    )["target"] = reply["id"]
    config["flow"]["edges"] += [
        edge(ask, action, "success"),
        edge(ask, retry, "invalid"),
        edge(ask, end, "exhausted"),
        edge(retry, ask),
        edge(reply, switch),
    ]
    action["config"]["mock_response"]["internal"] = "private-response"
    integration = MockIntegration(
        id=UUID(action["config"]["integration_id"]),
        enabled=True,
        methods=["POST"],
        paths=["/orders"],
        response_fields=["status"],
    )
    version = uuid4()

    def run(allowed):
        engine = RuntimeEngine(
            version, config, lambda _: Prediction(version, None), mock_integrations=allowed
        )
        opened = engine.start()
        assert opened.messages == ("Укажите номер заказа",)
        invalid = engine.send(opened.state, "01")
        assert invalid.messages == ("Нужны пять цифр", "Укажите номер заказа")
        assert not any(event.node_type == "http_action" for event in invalid.trace)
        return engine.send(invalid.state, "12345")

    result = run([integration])
    assert result.messages == ("Статус: готов",) and result.state.status == "waiting_input"
    assert "12345" not in repr(result.trace) and "готов" not in repr(result.trace)
    assert dict(result.state.slot_values) == {"order": "12345", "status": "готов"}
    rejected = run([integration.model_copy(update={"response_fields": []})])
    assert any(event.output == "invalid_response" for event in rejected.trace)
    assert "status" not in dict(rejected.state.slot_values)
    denied = run([])
    assert any(event.output == "http_error" for event in denied.trace)
    action["config"]["mock_outcome"] = "timeout"
    assert any(event.output == "timeout" for event in run([integration]).trace)


def test_mock_action_turn_is_limited_to_three_calls():
    config, action = configuration()
    config["slots"][0]["default"] = "order"
    flow = config["flow"]
    flow["edges"][0]["target"] = action["id"]
    previous = action
    end = flow["nodes"][3]
    for _ in range(3):
        current = deepcopy(action)
        current["id"] = str(uuid4())
        flow["nodes"].append(current)
        next(
            e
            for e in flow["edges"]
            if e["source"] == previous["id"] and e["source_handle"] == "success"
        )["target"] = current["id"]
        flow["edges"] += [
            {
                "id": str(uuid4()),
                "source": current["id"],
                "target": end["id"],
                "source_handle": handle,
            }
            for handle in ("success", "http_error", "timeout", "invalid_response", "unknown_result")
        ]
        previous = current
    integration = MockIntegration(
        id=UUID(action["config"]["integration_id"]),
        enabled=True,
        methods=["POST"],
        paths=["/orders"],
        response_fields=["status"],
    )
    version = uuid4()
    result = RuntimeEngine(
        version, config, lambda _: Prediction(version, None), mock_integrations=[integration]
    ).start()
    assert result.state.status == "failed" and result.error_code == "http_action_limit"
