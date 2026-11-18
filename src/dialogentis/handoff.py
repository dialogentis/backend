"""Explicit parameter mappings between pinned agent versions."""

from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field, StrictBool, ValidationError, model_validator

from dialogentis.contracts import InputModel
from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.slots import Slot

MAX_HANDOFF_DEPTH = 5


class HandoffDraftConfig(InputModel):
    model_config = ConfigDict(frozen=True)

    target_agent_id: UUID | None = None
    target_version_id: UUID | None = None
    mode: Literal["call", "transfer"] = "call"
    input_mapping: dict[UUID, UUID] = Field(default_factory=dict, max_length=100)
    return_mapping: dict[UUID, UUID] = Field(default_factory=dict, max_length=100)
    include_history: StrictBool = False
    forward_current_message: StrictBool = False

    @model_validator(mode="after")
    def validate_mappings(self):
        for mapping in (self.input_mapping, self.return_mapping):
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("Each destination parameter may be assigned once")
        if self.mode == "transfer" and self.return_mapping:
            raise ValueError("Transfer has no return mapping")
        return self


class HandoffConfig(HandoffDraftConfig):
    target_agent_id: UUID
    target_version_id: UUID


class ReturnConfig(InputModel):
    slot_ids: list[UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_slots(self):
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("Duplicate return parameter")
        return self


def handoff_outputs(mode: str) -> tuple[str, ...]:
    return ("returned", "error") if mode == "call" else ("error",)


def check_mapping(mapping: dict[UUID, UUID], sources: dict[UUID, Slot], targets: dict[UUID, Slot]):
    for source_id, target_id in mapping.items():
        source, target = sources.get(source_id), targets.get(target_id)
        if (
            source is None
            or target is None
            or source.value_type != target.value_type
            or (source.sensitive and not target.sensitive)
        ):
            raise ValueError("Incompatible handoff parameters")


def validate_handoff_references(flow: Flow, slots: list[Slot], *, complete: bool = False):
    slot_ids = {slot.id for slot in slots}
    issues: list[Issue] = []
    for index, node in enumerate(flow.nodes):
        try:
            if node.type == "handoff":
                schema = HandoffConfig if complete else HandoffDraftConfig
                config = schema.model_validate(node.config)
                if (
                    set(config.input_mapping) - slot_ids
                    or set(config.return_mapping.values()) - slot_ids
                ):
                    raise ValueError("Unknown local handoff parameter")
            elif node.type == "return":
                if set(ReturnConfig.model_validate(node.config).slot_ids) - slot_ids:
                    raise ValueError("Unknown local return parameter")
        except ValidationError, ValueError:
            issues.append(
                Issue(
                    ("flow", "nodes", index, "config"),
                    "invalid_handoff_config",
                    "Проверьте настройки передачи диалога и параметры блока",
                )
            )
    return issues
