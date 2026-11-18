from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CODE_POINTS = 16_000


class ContextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "agent"]
    text: str = Field(max_length=4000, repr=False)


class HistoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ContextMessage, ...] = Field(
        default=(), max_length=MAX_HISTORY_MESSAGES, repr=False
    )
    truncated: bool = False

    @model_validator(mode="after")
    def within_budget(self):
        if sum(len(message.text) for message in self.messages) > MAX_HISTORY_CODE_POINTS:
            raise ValueError("Invalid history length")
        return self


def recent_history(newest_first: Iterable[ContextMessage]) -> HistoryContext:
    remaining = MAX_HISTORY_CODE_POINTS
    messages = []
    truncated = False
    for message in newest_first:
        if len(messages) == MAX_HISTORY_MESSAGES or remaining == 0:
            truncated = True
            break
        if len(message.text) > remaining:
            messages.append(ContextMessage(role=message.role, text=message.text[-remaining:]))
            truncated = True
            break
        messages.append(message)
        remaining -= len(message.text)
    return HistoryContext(messages=tuple(reversed(messages)), truncated=truncated)
