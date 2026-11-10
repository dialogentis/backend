"""Parse a user's slot input without binary-float conversion or executable rules."""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

import re2
from pydantic import ValidationError

from dialogentis.ask_slot import DECIMAL, AskSlotConfig, LengthCheck, PatternCheck, RangeCheck
from dialogentis.slots import Slot


class SlotValueError(ValueError):
    def __init__(self, code: str = "invalid_slot_value"):
        self.code = code
        super().__init__(code)


def valid_value(slot: Slot, value: object, config: AskSlotConfig) -> bool:
    if value is None or (slot.required and value == ""):
        return False
    try:
        Slot.model_validate({**slot.model_dump(), "default": value})
    except ValidationError:
        return False
    for check in config.checks:
        if isinstance(check, LengthCheck):
            if not isinstance(value, str):
                return False
            if check.kind == "min_length" and len(value) < check.value:
                return False
            if check.kind == "max_length" and len(value) > check.value:
                return False
        elif isinstance(check, RangeCheck):
            if type(value) is not int and not isinstance(value, str):
                return False
            try:
                numeric = Decimal(str(value))
            except ValueError, InvalidOperation:
                return False
            bound = Decimal(check.value)
            if (check.kind == "minimum" and numeric < bound) or (
                check.kind == "maximum" and numeric > bound
            ):
                return False
        elif isinstance(check, PatternCheck):
            if not isinstance(value, str):
                return False
            options = re2.Options()
            options.log_errors = False
            options.max_mem = 8 * 1024 * 1024
            if re2.compile(check.pattern, options=options).fullmatch(value) is None:
                return False
    return True


def parse_slot_value(slot: Slot, text: str, config: AskSlotConfig) -> str | int | bool:
    if not isinstance(text, str) or len(text) > 2000:
        raise SlotValueError()
    try:
        text.encode("utf-8")
    except UnicodeError:
        raise SlotValueError() from None
    value = text.strip()
    if not value:
        raise SlotValueError()
    if slot.value_type == "string":
        result: str | int | bool = value
    elif slot.value_type == "integer":
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise SlotValueError()
        result = int(value)
        if abs(result) > 2**53 - 1:
            raise SlotValueError()
    elif slot.value_type == "number":
        if len(value) > 128 or not DECIMAL.fullmatch(value):
            raise SlotValueError()
        result = value
    elif slot.value_type == "boolean":
        if value.casefold() not in {"true", "false", "да", "нет"}:
            raise SlotValueError()
        result = value.casefold() in {"true", "да"}
    else:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Invalid date")
        except ValueError:
            raise SlotValueError() from None
        result = value
    if not valid_value(slot, result, config):
        raise SlotValueError()
    return result
