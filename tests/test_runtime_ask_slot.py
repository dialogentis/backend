from uuid import UUID, uuid4

import pytest

from dialogentis.agents import empty_config
from dialogentis.nlu.dialogue import DialogueDecision
from dialogentis.runtime.core import (
    RuntimeEngine,
    RuntimeFault,
    RuntimeState,
)
from tests.dialogue_fixtures import clarify, route, value
from tests.test_runtime_core import edge, node


def ask_config(value_type="integer", **ask_values):
    config = empty_config()
    slot_id = uuid4()
    config["slots"] = [
        {
            "id": str(slot_id),
            "code": "order_id",
            "value_type": value_type,
            "required": True,
            "sensitive": True,
        }
    ]
    start, switch, fallback, end = config["flow"]["nodes"]
    ask = node(
        "ask_slot",
        slot_id=str(slot_id),
        prompt="Укажите значение",
        max_attempts=2,
        checks=[],
        **ask_values,
    )
    retry = node("say", text="Значение не подошло")
    response = node("say", text="Получено: {{order_id}}")
    end["config"]["text"] = "Диалог завершён"
    config["flow"]["nodes"] += [ask, retry, response]
    config["flow"]["edges"][0]["target"] = ask["id"]
    config["flow"]["edges"] += [
        edge(ask, response, "success"),
        edge(response, switch),
        edge(ask, retry, "invalid"),
        edge(retry, ask),
        edge(ask, end, "exhausted"),
    ]
    return config, ask, slot_id


def test_prompt_retry_success_and_no_input_reuse():
    version = uuid4()
    config, ask, _ = ask_config()
    calls = []
    engine = RuntimeEngine(
        version,
        config,
        lambda message: calls.append(message.text) or clarify(message),
    )
    first = engine.start()
    assert first.messages == ("Укажите значение",) and first.state.pending_ask == UUID(ask["id"])
    invalid = engine.send(first.state, "не число")
    assert invalid.messages == ("Значение не подошло", "Укажите значение")
    assert invalid.state.attempts == ((UUID(ask["id"]), 1),)
    valid = engine.send(RuntimeState.model_validate_json(invalid.state.model_dump_json()), "92817")
    assert valid.messages == ("Получено: 92817",)
    assert valid.state.pending_ask is None and valid.state.attempts == ()
    assert dict(valid.state.slot_values)["order_id"] == 92817
    assert valid.state.status == "waiting_input" and calls == []
    assert "92817" not in repr(valid.trace) and "92817" not in repr(valid.state)


def test_exhausted_retries_clear_values_and_checkpoint():
    version = uuid4()
    config, _, _ = ask_config()
    engine = RuntimeEngine(version, config, clarify)
    first = engine.send(engine.start().state, "первый неверный")
    result = engine.send(first.state, "второй неверный")
    assert result.state.status == "completed" and result.messages == ("Диалог завершён",)
    assert result.state.slot_values == () and result.state.attempts == ()
    assert "второй неверный" not in repr(result.trace)


def test_successful_slot_input_resets_consecutive_fallbacks():
    version = uuid4()
    config, _, _ = ask_config()
    engine = RuntimeEngine(version, config, clarify)
    waiting = engine.start().state.model_copy(update={"fallback_count": 2})
    result = engine.send(waiting, "42")
    assert result.state.status == "waiting_input" and result.state.fallback_count == 0


def test_valid_default_is_not_treated_as_missing_when_false_or_zero():
    version = uuid4()
    for kind, default, displayed in (("integer", 0, "0"), ("boolean", False, "нет")):
        config, _, _ = ask_config(kind)
        config["slots"][0]["default"] = default
        engine = RuntimeEngine(version, config, clarify)
        result = engine.start()
        assert result.messages == (f"Получено: {displayed}",)
        assert (
            result.state.pending_ask is None
            and dict(result.state.slot_values)["order_id"] == default
        )


def test_entity_from_intent_message_is_exact_and_used_once():
    version, intent_id, entity_id = uuid4(), uuid4(), uuid4()
    config, ask, _ = ask_config("number", entity_id=str(entity_id))
    config["slots"][0]["sensitive"] = False
    config["intents"] = [{"id": str(intent_id), "code": "order", "name": "Заказ"}]
    config["entities"] = [
        {
            "id": str(entity_id),
            "code": "amount",
            "name": "Сумма",
            "value_type": "number",
            "method": "dictionary",
            "dictionary": [{"value": "12345678901234567890.01", "synonyms": ["сумма"]}],
        }
    ]
    start, switch = config["flow"]["nodes"][:2]
    config["flow"]["edges"][0]["target"] = switch["id"]
    config["flow"]["edges"].append(edge(switch, ask, str(intent_id)))
    calls = []

    def predict(message):
        calls.append(message.text)
        return route("order", value("order_id", "сумма"))

    engine = RuntimeEngine(version, config, predict)
    result = engine.send(engine.start().state, "заказ сумма")
    assert result.messages == ("Получено: 12345678901234567890.01",)
    assert dict(result.state.slot_values)["order_id"] == "12345678901234567890.01"
    assert result.state.status == "waiting_input" and calls == ["заказ сумма"]
    assert all("сумма" not in repr(event) for event in result.trace)
    assert start["type"] == "start"


def test_entity_value_can_be_taken_from_an_answer_and_bad_spans_are_rejected():
    version, entity_id = uuid4(), uuid4()
    config, _, _ = ask_config("string", entity_id=str(entity_id))
    config["entities"] = [
        {
            "id": str(entity_id),
            "code": "city",
            "name": "Город",
            "value_type": "string",
            "method": "dictionary",
            "dictionary": [{"value": "Москва", "synonyms": ["столица"]}],
        }
    ]

    config["slots"][0]["sensitive"] = False

    def predict(_):
        return DialogueDecision(
            action="continue", intent=None, values=(value("order_id", "столица"),)
        )

    engine = RuntimeEngine(version, config, predict)
    result = engine.send(engine.start().state, "в столицу столица")
    assert result.messages == ("Получено: Москва",)
    assert dict(result.state.slot_values)["order_id"] == "Москва"
    bad = RuntimeEngine(
        version,
        config,
        lambda _: DialogueDecision(
            action="continue", intent=None, values=(value("order_id", "SECRET"),)
        ),
    )
    result = bad.send(bad.start().state, "столица")
    assert result.error_code == "nlu_error" and "SECRET" not in repr(result.trace)
    assert result.state.slot_values == ()


def test_multiple_explicit_entities_fill_their_own_slots():
    version, intent_id, city_id, product_id, second_slot = (uuid4() for _ in range(5))
    config, first_ask, _ = ask_config("string", entity_id=str(city_id))
    config["slots"][0]["sensitive"] = False
    config["intents"] = [{"id": str(intent_id), "code": "order", "name": "Заказ"}]
    config["entities"] = [
        {
            "id": str(city_id),
            "code": "city",
            "name": "Город",
            "value_type": "string",
            "method": "dictionary",
            "dictionary": [{"value": "Омск"}],
        },
        {
            "id": str(product_id),
            "code": "item",
            "name": "Товар",
            "value_type": "string",
            "method": "dictionary",
            "dictionary": [{"value": "Книга"}],
        },
    ]
    config["slots"].append({"id": str(second_slot), "code": "item", "value_type": "string"})
    switch, end = config["flow"]["nodes"][1], config["flow"]["nodes"][3]
    second_ask = node(
        "ask_slot",
        slot_id=str(second_slot),
        prompt="Какой товар?",
        entity_id=str(product_id),
        max_attempts=2,
        checks=[],
    )
    config["flow"]["nodes"].append(second_ask)
    config["flow"]["edges"][0]["target"] = switch["id"]
    config["flow"]["edges"] += [
        edge(switch, first_ask, str(intent_id)),
        edge(second_ask, switch, "success"),
        edge(second_ask, second_ask, "invalid"),
        edge(second_ask, end, "exhausted"),
    ]
    next(
        item
        for item in config["flow"]["edges"]
        if item["source"] == first_ask["id"] and item["source_handle"] == "success"
    )["target"] = second_ask["id"]

    engine = RuntimeEngine(
        version, config, lambda _: route("order", value("order_id", "Омск"), value("item", "Книга"))
    )
    result = engine.send(engine.start().state, "Омск Книга")
    assert result.messages == ()
    assert result.state.pending_ask is None
    assert dict(result.state.slot_values) == {"order_id": "Омск", "item": "Книга"}


def test_ambiguous_entities_count_as_invalid_and_checkpoint_is_checked():
    version, entity_id = uuid4(), uuid4()
    config, ask, _ = ask_config("string", entity_id=str(entity_id))
    config["entities"] = [
        {
            "id": str(entity_id),
            "code": "city",
            "name": "Город",
            "value_type": "string",
            "method": "dictionary",
            "dictionary": [{"value": "Омск"}, {"value": "Томск"}],
        }
    ]
    config["slots"][0]["sensitive"] = False
    engine = RuntimeEngine(version, config, clarify)
    waiting = engine.start().state
    result = engine.send(waiting, "Омск Томск")
    assert result.messages == ("Значение не подошло", "Укажите значение")
    assert result.state.attempts == ((UUID(ask["id"]), 1),)
    broken = waiting.model_copy(update={"pending_ask": uuid4()})
    with pytest.raises(RuntimeFault, match="invalid_checkpoint"):
        engine.send(broken, "Омск")
