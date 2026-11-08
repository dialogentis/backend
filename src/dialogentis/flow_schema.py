from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue

from dialogentis.contracts import InputModel


class Position(InputModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class FlowNode(InputModel):
    id: UUID
    type: Literal[
        "start",
        "switch_intent",
        "say",
        "ask_slot",
        "set_slot",
        "condition",
        "http_action",
        "handoff",
        "return",
        "fallback",
        "end",
    ]
    config: dict[str, JsonValue]
    position: Position


class FlowEdge(InputModel):
    id: UUID
    source: UUID
    source_handle: str = Field(min_length=1, max_length=128)
    target: UUID


class Flow(InputModel):
    schema_version: Literal[1]
    flow_id: UUID
    nodes: list[FlowNode] = Field(max_length=300)
    edges: list[FlowEdge] = Field(max_length=600)
    layout: dict[str, JsonValue]
