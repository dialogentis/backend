"""Declarative slot assignment; values never contain executable expressions."""

from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, JsonValue, ValidationError

from dialogentis.contracts import InputModel
from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.slots import Slot


class SetSlotDraftConfig(InputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot_id: UUID | None = None
    operation: Literal["set", "clear"] = "set"
    source: Literal["constant", "slot"] = "constant"
    source_slot_id: UUID | None = None
    value: JsonValue = None


class SetSlotConfig(SetSlotDraftConfig):
    slot_id: UUID


def check_set_config(config: SetSlotConfig, slots: dict[UUID, Slot]) -> None:
    target = slots.get(config.slot_id)
    if target is None:
        raise ValueError("Unknown target")
    if config.operation == "clear":
        return
    if config.source == "slot":
        source = slots.get(config.source_slot_id)
        if source is None or source.value_type != target.value_type:
            raise ValueError("Unknown or incompatible source")
    else:
        if config.value is None:
            raise ValueError("Missing value")
        try:
            if isinstance(config.value, str):
                config.value.encode("utf-8")
            Slot.model_validate({**target.model_dump(), "default": config.value})
        except ValidationError, UnicodeError:
            raise ValueError("Invalid value") from None


def validate_set_references(flow: Flow, slots: list[Slot]) -> list[Issue]:
    ids = {slot.id for slot in slots}
    issues = []
    for index, node in enumerate(flow.nodes):
        if node.type != "set_slot":
            continue
        for field in ("slot_id", "source_slot_id"):
            raw = node.config.get(field)
            if raw is None:
                continue
            try:
                reference = UUID(raw) if isinstance(raw, str) else None
            except ValueError:
                reference = None
            if reference not in ids:
                issues.append(
                    Issue(
                        ("flow", "nodes", index, "config", field),
                        "unknown_set_slot",
                        "Параметр блока присваивания не найден",
                    )
                )
    return issues
