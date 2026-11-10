"""Slot definitions shared by draft validation, import and copying."""

import re
from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue, StrictBool, ValidationError, model_validator

from dialogentis.contracts import InputModel
from dialogentis.datasets.validation import Issue


class Slot(InputModel):
    id: UUID
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(default="", max_length=120)
    value_type: Literal["string", "integer", "number", "boolean", "date"] = "string"
    required: StrictBool = False
    sensitive: StrictBool = False
    default: JsonValue = None

    @model_validator(mode="after")
    def check_default(self):
        value = self.default
        if value is None:
            return self
        if self.value_type == "string" and isinstance(value, str) and len(value) <= 2000:
            return self
        if self.value_type == "integer" and type(value) is int and abs(value) <= 2**53 - 1:
            return self
        if self.value_type == "number" and isinstance(value, str) and len(value) <= 128:
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value):
                return self
        if self.value_type == "boolean" and type(value) is bool:
            return self
        if self.value_type == "date" and isinstance(value, str):
            try:
                if date.fromisoformat(value).isoformat() == value:
                    return self
            except ValueError:
                pass
        raise ValueError("Invalid default for slot type")


def validate_slots(raw: object) -> tuple[list[Slot], list[Issue]]:
    if not isinstance(raw, list):
        return [], [Issue(("slots",), "invalid_slots", "Некорректный список параметров")]
    slots: list[Slot] = []
    issues: list[Issue] = []
    ids: dict[UUID, int] = {}
    codes: dict[str, int] = {}
    for index, item in enumerate(raw):
        try:
            slot = Slot.model_validate(item)
        except ValidationError as exc:
            for error in exc.errors(include_input=False, include_context=False, include_url=False):
                field = (
                    error["loc"][0]
                    if error["loc"] and error["loc"][0] in Slot.model_fields
                    else None
                )
                path = ("slots", index, field) if field else ("slots", index)
                issues.append(Issue(path, "invalid_slot", "Проверьте тип и значение параметра"))
            continue
        if slot.id in ids:
            issues.append(
                Issue(
                    ("slots", index, "id"),
                    "duplicate_slot_id",
                    "Идентификатор параметра повторяется",
                    ids[slot.id],
                )
            )
        else:
            ids[slot.id] = index
        if slot.code in codes:
            issues.append(
                Issue(
                    ("slots", index, "code"),
                    "duplicate_slot_code",
                    "Код параметра уже используется",
                    codes[slot.code],
                )
            )
        else:
            codes[slot.code] = index
        slots.append(slot)
    return slots, issues


def removed_slot_references(previous: dict, current: dict) -> list[Issue]:
    old_slots, _ = validate_slots(previous.get("slots", []))
    new_slots, _ = validate_slots(current.get("slots", []))
    new_by_id = {slot.id: slot for slot in new_slots}
    removed_ids = {str(slot.id) for slot in old_slots if slot.id not in new_by_id}
    removed_codes = {
        slot.code
        for slot in old_slots
        if slot.id not in new_by_id or new_by_id[slot.id].code != slot.code
    }
    if not removed_ids and not removed_codes:
        return []

    def references(value: object) -> bool:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                if item in removed_ids or any(
                    match.group(1) in removed_codes
                    for match in re.finditer(r"\{\{\s*([a-z][a-z0-9_]{1,63})\s*\}\}", item)
                ):
                    return True
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
        return False

    flow = current.get("flow")
    nodes = flow.get("nodes", []) if isinstance(flow, dict) else []
    if not isinstance(nodes, list):
        return []
    return [
        Issue(
            ("flow", "nodes", index, "config"),
            "slot_in_use",
            "Сначала удалите ссылки на изменяемый параметр",
        )
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and references(node.get("config"))
    ]
