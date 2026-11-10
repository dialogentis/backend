from uuid import uuid4

import pytest
from pydantic import ValidationError

from dialogentis.ask_slot import AskSlotConfig, check_ask_config
from dialogentis.runtime.slot_values import SlotValueError, parse_slot_value, valid_value
from dialogentis.slots import Slot


def slot(value_type, **values):
    return Slot.model_validate(
        {"id": str(uuid4()), "code": "value", "value_type": value_type, **values}
    )


def ask(slot, **values):
    return AskSlotConfig.model_validate(
        {"slot_id": str(slot.id), "prompt": "Укажите значение", **values}
    )


@pytest.mark.parametrize(
    ("kind", "text", "expected"),
    [
        ("string", "  Москва 👩🏽‍💻  ", "Москва 👩🏽‍💻"),
        ("integer", "+42", 42),
        ("number", "12345678901234567890.0001", "12345678901234567890.0001"),
        ("boolean", "ДА", True),
        ("boolean", "нет", False),
        ("date", "2024-02-29", "2024-02-29"),
    ],
)
def test_parse_values_without_loss(kind, text, expected):
    definition = slot(kind)
    assert parse_slot_value(definition, text, ask(definition)) == expected


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("string", "   "),
        ("integer", "1.0"),
        ("integer", str(2**53)),
        ("number", "1e4"),
        ("number", "NaN"),
        ("number", "+1"),
        ("boolean", "может быть"),
        ("date", "01.02.2024"),
        ("date", "2023-02-29"),
        ("string", "\ud800"),
    ],
)
def test_invalid_values_have_safe_errors(kind, text):
    definition = slot(kind, sensitive=True)
    with pytest.raises(SlotValueError) as error:
        parse_slot_value(definition, text, ask(definition))
    assert error.value.code == "invalid_slot_value" and text not in str(error.value)


def test_checks_are_typed_bounded_and_use_safe_regex():
    text_slot = slot("string")
    config = ask(
        text_slot,
        checks=[
            {"kind": "min_length", "value": 2},
            {"kind": "max_length", "value": 5},
            {"kind": "pattern", "pattern": "[А-Яа-я]+"},
        ],
    )
    check_ask_config(config, text_slot, [text_slot])
    assert parse_slot_value(text_slot, "Омск", config) == "Омск"
    for text in ("Я", "Новосибирск", "42"):
        with pytest.raises(SlotValueError):
            parse_slot_value(text_slot, text, config)
    number = slot("number")
    config = ask(
        number,
        checks=[{"kind": "minimum", "value": "0.01"}, {"kind": "maximum", "value": "100.50"}],
    )
    check_ask_config(config, number, [number])
    assert parse_slot_value(number, "100.50", config) == "100.50"
    with pytest.raises(SlotValueError):
        parse_slot_value(number, "100.5001", config)
    with pytest.raises(ValueError):
        check_ask_config(config, text_slot, [text_slot])
    with pytest.raises(ValidationError):
        ask(text_slot, checks=[{"kind": "pattern", "pattern": r"(a)\1"}])
    with pytest.raises(ValidationError):
        ask(
            text_slot,
            checks=[{"kind": "min_length", "value": 9}, {"kind": "max_length", "value": 3}],
        )


def test_required_empty_string_is_not_a_collected_value():
    definition = slot("string", required=True)
    assert not valid_value(definition, "", ask(definition))
    boolean = slot("boolean", required=True)
    assert valid_value(boolean, False, ask(boolean))


def test_ask_prompt_cannot_require_the_value_before_it_is_collected():
    definition = slot("string", required=True)
    with pytest.raises(ValueError):
        check_ask_config(ask(definition, prompt="Введите {{value}}"), definition, [definition])
