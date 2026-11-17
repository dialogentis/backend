"""Declarative HTTP actions; credentials and destinations belong to integrations."""

import json
from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, StrictStr, ValidationError, field_validator

from dialogentis.contracts import InputModel
from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.integrations.schema import Method, safe_path
from dialogentis.slots import Slot

HTTP_OUTPUTS = ("success", "http_error", "timeout", "invalid_response", "unknown_result")


class HttpActionDraftConfig(InputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    integration_id: UUID | None = None
    method: Method = "GET"
    path: StrictStr = ""
    query: dict[str, UUID] = Field(default_factory=dict, max_length=100)
    body: dict[str, UUID] = Field(default_factory=dict, max_length=100)
    response_mapping: dict[str, UUID] = Field(default_factory=dict, max_length=100)
    mock_outcome: Literal[
        "success", "http_error", "timeout", "invalid_response", "unknown_result"
    ] = "success"
    mock_response: JsonValue = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return safe_path(value) if value else value

    @field_validator("query", "body", "response_mapping")
    @classmethod
    def validate_names(cls, value: dict[str, UUID]) -> dict[str, UUID]:
        if any(
            not key or len(key) > 128 or any(ord(c) < 32 or ord(c) == 127 for c in key)
            for key in value
        ):
            raise ValueError("Invalid mapping name")
        return value

    @field_validator("mock_response")
    @classmethod
    def validate_mock(cls, value: JsonValue) -> JsonValue:
        try:
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except ValueError, UnicodeError, RecursionError:
            raise ValueError("Invalid mock response") from None
        if len(content) > 256 * 1024:
            raise ValueError("Mock response too large")
        return value


class HttpActionConfig(HttpActionDraftConfig):
    integration_id: UUID
    path: StrictStr = Field(min_length=1)


def check_http_references(config: HttpActionDraftConfig, slots: dict[UUID, Slot]) -> None:
    if any(
        ref not in slots
        for mapping in (config.query, config.body, config.response_mapping)
        for ref in mapping.values()
    ):
        raise ValueError("Unknown slot")
    if len(set(config.response_mapping.values())) != len(config.response_mapping):
        raise ValueError("Duplicate response target")


def validate_http_references(flow: Flow, slots: list[Slot]) -> list[Issue]:
    issues = []
    for index, node in enumerate(flow.nodes):
        if node.type != "http_action":
            continue
        try:
            action = HttpActionDraftConfig.model_validate(node.config)
            check_http_references(action, {slot.id: slot for slot in slots})
        except ValidationError, ValueError:
            issues.append(
                Issue(
                    ("flow", "nodes", index, "config"),
                    "invalid_http_action",
                    "Проверьте настройки HTTP-запроса и параметры",
                )
            )
    return issues
