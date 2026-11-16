"""Bounded declarative conditions for scenario branches."""

from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import Field, JsonValue, ValidationError

from dialogentis.contracts import InputModel
from dialogentis.slots import Slot


class Branch(InputModel):
    id: UUID
    expression: dict[str, JsonValue]


class ConditionConfig(InputModel):
    branches: list[Branch] = Field(min_length=1, max_length=50)


def validate_expression(expression: object, slots: dict[UUID, Slot], depth: int = 1) -> None:
    if depth > 3 or not isinstance(expression, dict):
        raise ValueError("Invalid condition depth or expression")
    kind = expression.get("op")
    if kind in ("all", "any"):
        children = expression.get("conditions")
        if (
            set(expression) != {"op", "conditions"}
            or not isinstance(children, list)
            or not 1 <= len(children) <= 20
        ):
            raise ValueError("Invalid composition")
        for child in children:
            validate_expression(child, slots, depth + 1)
        return
    if kind not in ("eq", "ne", "gt", "gte", "lt", "lte", "in", "exists"):
        raise ValueError("Unknown operator")
    expected = {"op", "slot_id"} if kind == "exists" else {"op", "slot_id", "value"}
    if set(expression) != expected or not isinstance(expression.get("slot_id"), str):
        raise ValueError("Invalid comparison")
    try:
        slot = slots[UUID(expression["slot_id"])]
    except ValueError, KeyError:
        raise ValueError("Unknown slot") from None
    if kind in ("gt", "gte", "lt", "lte") and slot.value_type not in ("integer", "number", "date"):
        raise ValueError("Unsupported ordering")
    if kind == "exists":
        return
    operand = expression["value"]
    values = operand if kind == "in" else [operand]
    if not isinstance(values, list) or not 1 <= len(values) <= 100:
        raise ValueError("Invalid operands")
    for value in values:
        if value is None or isinstance(value, (dict, list)):
            raise ValueError("Invalid operand")
        try:
            if isinstance(value, str):
                value.encode("utf-8")
            Slot.model_validate({**slot.model_dump(), "default": value})
        except ValidationError, UnicodeError:
            raise ValueError("Invalid operand type") from None


def validate_condition(config: ConditionConfig, slots: dict[UUID, Slot]) -> None:
    if len({branch.id for branch in config.branches}) != len(config.branches):
        raise ValueError("Duplicate branch")
    for branch in config.branches:
        validate_expression(branch.expression, slots)


def evaluate_expression(expression: dict, slots: dict[UUID, Slot], values: dict) -> bool:
    kind = expression["op"]
    if kind in ("all", "any"):
        results = (evaluate_expression(child, slots, values) for child in expression["conditions"])
        return all(results) if kind == "all" else any(results)
    slot = slots[UUID(expression["slot_id"])]
    value: Any = values.get(slot.code)
    exists = value is not None and value != ""
    if kind == "exists":
        return exists
    if not exists:
        return False
    operand: Any = expression["value"]
    if kind == "in":
        if slot.value_type == "number":
            return Decimal(value) in [Decimal(item) for item in operand]
        return value in operand
    if slot.value_type == "number":
        value = Decimal(value)
        operand = Decimal(operand)
    if kind == "eq":
        return value == operand
    if kind == "ne":
        return value != operand
    if kind == "gt":
        return value > operand
    if kind == "gte":
        return value >= operand
    if kind == "lt":
        return value < operand
    return value <= operand


def choose_branch(config: ConditionConfig, slots: dict[UUID, Slot], values: dict) -> str:
    for branch in config.branches:
        if evaluate_expression(branch.expression, slots, values):
            return str(branch.id)
    return "else"
