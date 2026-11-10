"""Declarative configuration and validation of a slot collection block."""

import re
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

import re2
from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from dialogentis.contracts import InputModel
from dialogentis.datasets.schema import Entity
from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.slots import Slot
from dialogentis.templates import parse_template

DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


class LengthCheck(InputModel):
    kind: Literal["min_length", "max_length"]
    value: StrictInt = Field(ge=0, le=2000)


class RangeCheck(InputModel):
    kind: Literal["minimum", "maximum"]
    value: str = Field(min_length=1, max_length=128)

    @field_validator("value")
    @classmethod
    def valid_decimal(cls, value: str) -> str:
        if not DECIMAL.fullmatch(value):
            raise ValueError("Expected a decimal string")
        return value


class PatternCheck(InputModel):
    kind: Literal["pattern"]
    pattern: str = Field(min_length=1, max_length=2048)

    @field_validator("pattern")
    @classmethod
    def safe_pattern(cls, pattern: str) -> str:
        options = re2.Options()
        options.log_errors = False
        options.max_mem = 8 * 1024 * 1024
        try:
            re2.compile(pattern, options=options)
        except re2.error, UnicodeError:
            raise ValueError("Unsupported pattern") from None
        return pattern


SlotCheck = Annotated[LengthCheck | RangeCheck | PatternCheck, Field(discriminator="kind")]


class AskSlotDraftConfig(InputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot_id: UUID | None = None
    prompt: str = Field(default="", max_length=4000)
    entity_id: UUID | None = None
    max_attempts: StrictInt = Field(default=3, ge=1, le=5)
    checks: list[SlotCheck] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def unique_checks(self):
        kinds = [check.kind for check in self.checks]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Duplicate check")
        lengths = {
            check.kind: check.value for check in self.checks if isinstance(check, LengthCheck)
        }
        if lengths.get("min_length", 0) > lengths.get("max_length", 2000):
            raise ValueError("Conflicting length checks")
        bounds = {
            check.kind: Decimal(check.value)
            for check in self.checks
            if isinstance(check, RangeCheck)
        }
        if "minimum" in bounds and "maximum" in bounds and bounds["minimum"] > bounds["maximum"]:
            raise ValueError("Conflicting numeric checks")
        return self


class AskSlotConfig(AskSlotDraftConfig):
    slot_id: UUID
    prompt: str = Field(min_length=1, max_length=4000)


def check_ask_config(config: AskSlotConfig, slot: Slot, slots: list[Slot]) -> None:
    if not config.prompt.strip():
        raise ValueError("Empty prompt")
    tokens = parse_template(config.prompt, slots)
    if slot.required and any(placeholder and token == slot.code for token, placeholder in tokens):
        raise ValueError("The prompt cannot require the value it is collecting")
    for check in config.checks:
        if isinstance(check, (LengthCheck, PatternCheck)) and slot.value_type != "string":
            raise ValueError("Text check requires a text slot")
        if isinstance(check, RangeCheck) and slot.value_type not in {"integer", "number"}:
            raise ValueError("Numeric check requires a numeric slot")
        if isinstance(check, RangeCheck) and slot.value_type == "integer" and "." in check.value:
            raise ValueError("Integer bound must be integral")


def validate_ask_references(flow: Flow, slots: list[Slot], entities: list[Entity]) -> list[Issue]:
    slot_ids, entity_ids = {slot.id for slot in slots}, {entity.id for entity in entities}
    issues = []
    for index, node in enumerate(flow.nodes):
        if node.type != "ask_slot":
            continue
        for field, valid_ids, code, message in (
            ("slot_id", slot_ids, "unknown_ask_slot", "Параметр для сбора не найден"),
            ("entity_id", entity_ids, "unknown_ask_entity", "Сущность для сбора не найдена"),
        ):
            raw = node.config.get(field)
            if raw is None:
                continue
            try:
                reference = UUID(raw) if isinstance(raw, str) else None
            except ValueError:
                reference = None
            if reference not in valid_ids:
                issues.append(Issue(("flow", "nodes", index, "config", field), code, message))
    return issues
