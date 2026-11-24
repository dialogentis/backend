from copy import deepcopy
from dataclasses import asdict
from uuid import UUID, uuid4

import pytest

from dialogentis.agents import empty_config
from dialogentis.runtime.core import (
    RuntimeEngine,
    RuntimeFault,
    RuntimeState,
)
from tests.dialogue_fixtures import clarify, route


def node(kind, **config):
    return {"id": str(uuid4()), "type": kind, "config": config, "position": {"x": 0, "y": 0}}


def edge(source, target, handle="next"):
    return {
        "id": str(uuid4()),
        "source": source["id"],
        "target": target["id"],
        "source_handle": handle,
    }


def conversation_config():
    config = empty_config()
    intent_id = uuid4()
    config["intents"] = [{"id": str(intent_id), "code": "order", "name": "Заказ"}]
    config["slots"] = [{"id": str(uuid4()), "code": "client", "required": True, "default": "Анна"}]
    start, switch, fallback, end = config["flow"]["nodes"]
    fallback["config"]["example"] = "Где мой заказ?"
    intro = node("say", text="Здравствуйте, {{client}}!")
    answer = node("say", text="Проверяю заказ.")
    end["config"]["text"] = "До свидания"
    config["flow"]["nodes"] += [intro, answer]
    config["flow"]["edges"][0]["target"] = intro["id"]
    config["flow"]["edges"] += [
        edge(intro, switch),
        edge(switch, answer, str(intent_id)),
        edge(answer, end),
    ]
    return config, intent_id


def test_multi_message_turns_and_version_pinned_checkpoints():
    version = uuid4()
    config, intent_id = conversation_config()
    calls = []

    def predict(message):
        calls.append(message.text)
        return route()

    engine = RuntimeEngine(version, config, predict)
    first = engine.start()
    assert first.state.status == "waiting_input" and first.state.turn == 0
    assert first.messages == ("Здравствуйте, Анна!",) and calls == []
    checkpoint = RuntimeState.model_validate_json(first.state.model_dump_json())
    config["flow"]["nodes"][2]["config"]["text"] = "Изменённая версия"
    result = engine.send(checkpoint, "PRIVATE_SENTINEL")
    assert result.state.status == "completed" and result.state.turn == 1
    assert result.messages == ("Проверяю заказ.", "До свидания")
    assert calls == ["PRIVATE_SENTINEL"] and "PRIVATE_SENTINEL" not in repr(result.trace)
    assert [event.event for event in result.trace] == ["transition", "transition", "complete"]
    with pytest.raises(RuntimeFault, match="conversation_finished"):
        engine.send(result.state, "ещё")
    changed = RuntimeEngine(version, config, predict)
    with pytest.raises(RuntimeFault, match="version_mismatch"):
        changed.send(checkpoint, "запрос")


def test_unknown_requests_wait_and_stop_after_three_with_explicit_example():
    version = uuid4()
    config, _ = conversation_config()
    engine = RuntimeEngine(version, config, clarify)
    result = engine.start()
    for attempt in (1, 2, 3):
        result = engine.send(result.state, "неизвестный запрос")
        assert result.state.fallback_count == attempt
        assert result.messages[0].startswith("Не удалось распознать")
        assert result.state.status == ("completed" if attempt == 3 else "waiting_input")
    assert result.messages[1] == "Например: Где мой заказ?"
    assert "заново" in result.messages[2]
    assert result.trace[-1].event == "fallback_limit"


def test_consumed_input_is_not_reused_and_known_request_resets_fallback():
    version = uuid4()
    config, intent_id = conversation_config()
    switch = config["flow"]["nodes"][1]
    answer = config["flow"]["nodes"][-1]
    config["flow"]["edges"][-1]["target"] = switch["id"]
    calls = []

    def predict(message):
        calls.append(message.text)
        return clarify(message) if len(calls) == 1 else route()

    engine = RuntimeEngine(version, config, predict)
    first = engine.send(engine.start().state, "неизвестно")
    second = engine.send(first.state, "где заказ")
    assert second.messages == ("Проверяю заказ.",)
    assert second.state.status == "waiting_input" and second.state.fallback_count == 0
    assert calls == ["неизвестно", "где заказ"]
    assert second.trace[-1].node_id == UUID(switch["id"]) and second.trace[-1].event == "wait"
    assert answer["type"] == "say"


def test_unsupported_blocks_and_missing_branches_fail_safely():
    version = uuid4()
    config, intent_id = conversation_config()
    config["flow"]["edges"] = [
        item for item in config["flow"]["edges"] if item["source_handle"] != str(intent_id)
    ]
    engine = RuntimeEngine(version, config, lambda _: route())
    result = engine.send(engine.start().state, "запрос")
    assert result.error_code == "nlu_error" and result.state.status == "failed"
    assert result.trace[-1].event == "error" and result.trace[-1].output == "nlu_error"
    assert "Попробуйте позже" in result.messages[-1]
    config = empty_config()
    start = config["flow"]["nodes"][0]
    unsupported = node("http_action")
    unsupported["config"] = {"integration_id": str(uuid4()), "path": "/orders"}
    config["flow"]["nodes"].append(unsupported)
    config["flow"]["edges"][0]["target"] = unsupported["id"]
    end = config["flow"]["nodes"][3]
    for handle in ("success", "http_error", "timeout", "invalid_response", "unknown_result"):
        config["flow"]["edges"].append(edge(unsupported, end, handle))
    result = RuntimeEngine(version, config, clarify).start()
    assert result.state.status == "failed" and result.error_code == "unsupported_in_stage"
    assert start["type"] == "start"


def test_invalid_nlu_or_slot_values_do_not_expose_private_input():
    version = uuid4()
    config, intent_id = conversation_config()
    config["slots"][0]["value_type"] = "integer"
    config["slots"][0]["default"] = None
    engine = RuntimeEngine(version, config, lambda _: route("unavailable"))
    with pytest.raises(RuntimeFault, match="invalid_slot_values") as invalid:
        engine.start({"client": "PRIVATE_SENTINEL"})
    assert "PRIVATE_SENTINEL" not in str(invalid.value)
    waiting = engine.start({"client": 5})
    result = engine.send(waiting.state, "PRIVATE_SENTINEL")
    assert result.error_code == "nlu_error"
    assert "PRIVATE_SENTINEL" not in repr(asdict(result))
    with pytest.raises(RuntimeFault, match="invalid_message"):
        engine.send(waiting.state, "")


def test_nlu_failure_and_invalid_results_are_safe():
    version = uuid4()
    config, intent_id = conversation_config()

    def broken(message):
        raise ValueError(f"PRIVATE_SENTINEL {message.text}")

    engine = RuntimeEngine(version, config, broken)
    result = engine.send(engine.start().state, "PRIVATE_SENTINEL")
    assert result.error_code == "nlu_error" and "PRIVATE_SENTINEL" not in repr(asdict(result))
    engine = RuntimeEngine(version, config, lambda _: route("unavailable"))
    assert engine.send(engine.start().state, "запрос").error_code == "nlu_error"


def test_transition_and_time_limits(monkeypatch):
    config = empty_config()
    start, end = config["flow"]["nodes"][0], config["flow"]["nodes"][3]
    responses = [node("say", text="Ответ") for _ in range(51)]
    config["flow"]["nodes"] = [start, *responses, end]
    chain = [start, *responses, end]
    config["flow"]["edges"] = [edge(a, b) for a, b in zip(chain, chain[1:], strict=False)]
    version = uuid4()
    engine = RuntimeEngine(version, config, clarify)
    assert engine.start().error_code == "transition_limit"
    ticks = iter([0.0, 21.0])
    monkeypatch.setattr("dialogentis.runtime.core.monotonic", lambda: next(ticks))
    assert engine.start().error_code == "turn_timeout"


def test_hash_ignores_layout_but_rejects_other_graph_or_version():
    version = uuid4()
    config, intent_id = conversation_config()
    engine = RuntimeEngine(version, config, lambda _: route())
    state = engine.start().state
    changed = deepcopy(config)
    changed["flow"]["layout"] = {"zoom": 2}
    changed["flow"]["nodes"][0]["position"] = {"x": 10, "y": 20}
    same = RuntimeEngine(version, changed, lambda _: route())
    assert same.send(state, "запрос").state.status == "completed"
    different = RuntimeEngine(uuid4(), config, lambda _: route())
    with pytest.raises(RuntimeFault, match="version_mismatch"):
        different.send(state, "запрос")
    changed["slots"][0]["name"] = "\ud800"
    with pytest.raises(RuntimeFault, match="invalid_flow"):
        RuntimeEngine(version, changed, lambda _: route())
