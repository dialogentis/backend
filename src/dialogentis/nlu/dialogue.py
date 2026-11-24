"""Contract between the scenario interpreter and the language model."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dialogentis.runtime.history_context import HistoryContext

type SlotText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class DialogueChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(max_length=64)
    name: str = Field(max_length=120)
    description: str = Field(max_length=1000)
    examples: tuple[SlotText, ...] = Field(default=(), max_length=2, repr=False)


class DialogueField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(max_length=64)
    name: str = Field(max_length=120)
    value_type: Literal["string", "integer", "number", "boolean", "date"]
    prompt: str = Field(max_length=500)
    entity: str | None = Field(default=None, max_length=120)


class DialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: UUID
    node_id: UUID
    text: str = Field(min_length=1, max_length=2000, repr=False)
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", repr=False)
    history: HistoryContext = Field(default_factory=HistoryContext, repr=False)
    choices: tuple[DialogueChoice, ...] = Field(max_length=100)
    fields: tuple[DialogueField, ...] = Field(max_length=32)
    awaiting: str | None = Field(default=None, max_length=64)


class DialogueValue(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    code: str = Field(max_length=64)
    value: SlotText = Field(repr=False)
    source: SlotText = Field(repr=False)


class DialogueDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    action: Literal["route", "continue", "clarify"]
    intent: str | None = Field(max_length=64)
    values: tuple[DialogueValue, ...] = Field(max_length=32, repr=False)

    @model_validator(mode="after")
    def validate_action(self):
        if (self.action == "route") != (self.intent is not None):
            raise ValueError("Only routing requires an intent")
        if self.action == "clarify" and self.values:
            raise ValueError("Clarification cannot supply values")
        return self


def accepts_decision(turn: DialogueTurn, decision: DialogueDecision) -> bool:
    if decision.action == "continue" and turn.awaiting is None:
        return False
    if decision.intent is not None and all(
        choice.code != decision.intent for choice in turn.choices
    ):
        return False
    codes = {value.code for value in decision.values}
    fields = {field.code for field in turn.fields}
    return (
        len(codes) == len(decision.values)
        and codes <= fields
        and all(
            value.source.strip()
            and value.source in turn.text
            and (turn.input_hash is None or "[скрыто]" not in value.source)
            for value in decision.values
        )
    )
