from uuid import uuid4

import pytest
from pydantic import ValidationError

from dialogentis.agents import empty_config
from dialogentis.slots import Slot, removed_slot_references, validate_slots


def item(**values):
    return {"id": str(uuid4()), "code": "parameter", **values}


@pytest.mark.parametrize(
    ("value_type", "default"),
    [
        ("string", "Москва"),
        ("integer", 2**53 - 1),
        ("number", "100.250"),
        ("boolean", False),
        ("date", "2024-02-29"),
        ("date", None),
    ],
)
def test_typed_slot_defaults_are_preserved(value_type, default):
    slot = Slot.model_validate(item(value_type=value_type, default=default))
    assert slot.default == default


@pytest.mark.parametrize(
    ("value_type", "default"),
    [
        ("string", 7),
        ("integer", True),
        ("integer", 2**53),
        ("number", 1.5),
        ("number", "NaN"),
        ("number", "1e3"),
        ("boolean", "false"),
        ("date", "2023-02-29"),
        ("date", "2024-2-9"),
    ],
)
def test_invalid_defaults_do_not_coerce(value_type, default):
    with pytest.raises(ValidationError):
        Slot.model_validate(item(value_type=value_type, default=default))


def test_duplicate_slots_and_invalid_values_do_not_echo_user_input():
    first = item(code="city")
    second = {**first, "name": "SECRET_SENTINEL"}
    third = item(code="invalid code", default="PRIVATE_SENTINEL")
    _, issues = validate_slots([first, second, third])
    assert {issue.code for issue in issues} == {
        "duplicate_slot_id",
        "duplicate_slot_code",
        "invalid_slot",
    }
    assert all(secret not in repr(issues) for secret in ("SECRET_SENTINEL", "PRIVATE_SENTINEL"))


def test_removing_or_renaming_referenced_slot_requires_clearing_references():
    previous = empty_config()
    slot = item(code="city")
    previous["slots"] = [slot]
    current = empty_config()
    current["flow"]["nodes"][2]["config"]["text"] = "Город: {{ city }}"
    assert removed_slot_references(previous, current)[0].code == "slot_in_use"
    current["slots"] = [{**slot, "code": "town"}]
    assert removed_slot_references(previous, current)[0].code == "slot_in_use"
    current["flow"]["nodes"][2]["config"]["text"] = "Город: {{town}}"
    assert not removed_slot_references(previous, current)
    current["slots"] = []
    current["flow"]["nodes"][2]["config"]["nested"] = {"slot_id": slot["id"]}
    assert removed_slot_references(previous, current)[0].code == "slot_in_use"
