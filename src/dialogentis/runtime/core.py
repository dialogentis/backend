"""Deterministic in-memory runtime; callers own persistence, isolation and timeouts."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

from dialogentis.ask_slot import AskSlotConfig
from dialogentis.datasets.schema import Entity, Intent
from dialogentis.datasets.validation import normalize_example
from dialogentis.flow_schema import Flow, FlowNode
from dialogentis.flow_validation import validate_flow
from dialogentis.runtime.slot_values import SlotValueError, parse_slot_value, valid_value
from dialogentis.slots import Slot, validate_slots
from dialogentis.templates import TemplateError, render_template

MAX_TRANSITIONS = 50
MAX_SECONDS = 20
FALLBACK_REASONS = {"empty_features", "low_score", "ambiguous_intent", "low_similarity"}
SAFE_FAILURE = "Не удалось выполнить сценарий. Попробуйте позже."


class RuntimeFault(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class EntityValue:
    entity_id: UUID
    start: int
    end: int
    text: str = field(repr=False)
    source: str = ""


@dataclass(frozen=True)
class Prediction:
    version_id: UUID
    intent_id: UUID | None
    fallback_reasons: tuple[str, ...] = ()
    entities: tuple[EntityValue, ...] = field(default=(), repr=False)


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version_id: UUID
    flow_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_id: UUID
    status: Literal["waiting_input", "completed", "failed"]
    turn: int = Field(ge=0)
    fallback_count: int = Field(ge=0, le=3)
    pending_ask: UUID | None = None
    slot_values: tuple[tuple[str, StrictBool | StrictInt | StrictStr | None], ...] = Field(
        default=(), repr=False
    )
    attempts: tuple[tuple[UUID, StrictInt], ...] = Field(default=(), repr=False)


@dataclass(frozen=True)
class TraceEvent:
    node_id: UUID
    node_type: str
    event: str
    output: str | None = None
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeResult:
    state: RuntimeState
    messages: tuple[str, ...]
    trace: tuple[TraceEvent, ...]
    error_code: str | None = None


def semantic_hash(
    flow: Flow, slots: list[Slot], intents: list[Intent], entities: list[Entity]
) -> str:
    payload = {
        "flow_id": str(flow.flow_id),
        "nodes": sorted(
            ({"id": str(node.id), "type": node.type, "config": node.config} for node in flow.nodes),
            key=lambda node: str(node["id"]),
        ),
        "edges": sorted(
            (
                {
                    "source": str(edge.source),
                    "target": str(edge.target),
                    "handle": edge.source_handle,
                }
                for edge in flow.edges
            ),
            key=lambda edge: (edge["source"], edge["handle"], edge["target"]),
        ),
        "slots": sorted(
            (slot.model_dump(mode="json") for slot in slots), key=lambda slot: slot["id"]
        ),
        "intents": sorted(
            ({"id": str(intent.id), "enabled": intent.enabled} for intent in intents),
            key=lambda intent: intent["id"],
        ),
        "entities": sorted(
            (entity.model_dump(mode="json") for entity in entities), key=lambda entity: entity["id"]
        ),
    }
    data = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class RuntimeEngine:
    def __init__(self, version_id: UUID, config: dict, predict: Callable[[str], Prediction]):
        errors, _ = validate_flow(config)
        if errors:
            raise RuntimeFault("invalid_flow")
        try:
            self._flow = Flow.model_validate(config["flow"]).model_copy(deep=True)
            self._slots, slot_errors = validate_slots(config.get("slots", []))
            self._intents = [Intent.model_validate(item) for item in config.get("intents", [])]
            self._entities = [Entity.model_validate(item) for item in config.get("entities", [])]
        except ValidationError, ValueError, TypeError, KeyError:
            raise RuntimeFault("invalid_flow") from None
        if (
            slot_errors
            or len({item.id for item in self._intents}) != len(self._intents)
            or len({item.id for item in self._entities}) != len(self._entities)
        ):
            raise RuntimeFault("invalid_flow")
        self._nodes = {node.id: node for node in self._flow.nodes}
        self._edges = {(edge.source, edge.source_handle): edge.target for edge in self._flow.edges}
        self._enabled = {intent.id for intent in self._intents if intent.enabled}
        self._slot_by_code = {slot.code: slot for slot in self._slots}
        self._slot_by_id = {slot.id: slot for slot in self._slots}
        self._entity_by_id = {entity.id: entity for entity in self._entities}
        self._asks = {
            node.id: AskSlotConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "ask_slot"
        }
        self._start = next(node.id for node in self._flow.nodes if node.type == "start")
        try:
            self._hash = semantic_hash(self._flow, self._slots, self._intents, self._entities)
        except TypeError, ValueError, UnicodeError:
            raise RuntimeFault("invalid_flow") from None
        self._version = version_id
        self._predict = predict

    def start(self, values: dict | None = None) -> RuntimeResult:
        if values is not None and not isinstance(values, dict):
            raise RuntimeFault("invalid_slot_values")
        initial = {slot.code: slot.default for slot in self._slots if slot.default is not None}
        initial.update(values or {})
        self._check_values(initial, "invalid_slot_values")
        return self._run(self._start, None, 0, 0, initial, {}, None)

    def send(self, state: RuntimeState, text: str, values: dict | None = None) -> RuntimeResult:
        if values is not None and not isinstance(values, dict):
            raise RuntimeFault("invalid_slot_values")
        if state.version_id != self._version or state.flow_hash != self._hash:
            raise RuntimeFault("version_mismatch")
        if state.status != "waiting_input":
            raise RuntimeFault("conversation_finished")
        if state.node_id not in self._nodes or state.fallback_count >= 3:
            raise RuntimeFault("invalid_checkpoint")
        if state.pending_ask is not None and (
            state.pending_ask != state.node_id or state.pending_ask not in self._asks
        ):
            raise RuntimeFault("invalid_checkpoint")
        stored = dict(state.slot_values)
        attempts = dict(state.attempts)
        if len(stored) != len(state.slot_values) or len(attempts) != len(state.attempts):
            raise RuntimeFault("invalid_checkpoint")
        if any(
            node not in self._asks or count < 1 or count >= self._asks[node].max_attempts
            for node, count in attempts.items()
        ):
            raise RuntimeFault("invalid_checkpoint")
        self._check_values(stored, "invalid_checkpoint")
        if not isinstance(text, str) or not 1 <= len(text) <= 2000:
            raise RuntimeFault("invalid_message")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise RuntimeFault("invalid_message") from None
        stored.update(values or {})
        self._check_values(stored, "invalid_slot_values")
        return self._run(
            state.node_id,
            text,
            state.turn + 1,
            state.fallback_count,
            stored,
            attempts,
            state.pending_ask,
        )

    def _check_values(self, values: dict, code: str) -> None:
        for key, value in values.items():
            slot = self._slot_by_code.get(key) if isinstance(key, str) else None
            if slot is None:
                raise RuntimeFault(code)
            if value is None:
                continue
            try:
                if isinstance(value, str):
                    value.encode("utf-8")
                Slot.model_validate({**slot.model_dump(), "default": value})
            except ValidationError, UnicodeError:
                raise RuntimeFault(code) from None

    def _run(
        self,
        node_id: UUID,
        text: str | None,
        turn: int,
        fallback_count: int,
        values: dict,
        attempts: dict[UUID, int],
        pending_ask: UUID | None,
    ) -> RuntimeResult:
        messages: list[str] = []
        trace: list[TraceEvent] = []
        transitions = 0
        deadline = monotonic() + MAX_SECONDS
        original_text = text
        last_prediction: Prediction | None = None
        entity_consumed = False

        def state(status: Literal["waiting_input", "completed", "failed"]) -> RuntimeState:
            return RuntimeState(
                version_id=self._version,
                flow_hash=self._hash,
                node_id=node_id,
                status=status,
                turn=turn,
                fallback_count=fallback_count,
                pending_ask=pending_ask,
                slot_values=tuple(sorted(values.items())),
                attempts=tuple((node, attempts[node]) for node in sorted(attempts, key=str)),
            )

        def fail(code: str) -> RuntimeResult:
            nonlocal pending_ask
            failed_node = self._nodes.get(node_id)
            trace.append(
                TraceEvent(node_id, failed_node.type if failed_node else "unknown", "error", code)
            )
            values.clear()
            attempts.clear()
            pending_ask = None
            messages.append(SAFE_FAILURE)
            return RuntimeResult(state("failed"), tuple(messages), tuple(trace), code)

        def move(node: FlowNode, handle: str) -> UUID:
            nonlocal transitions
            target = self._edges.get((node.id, handle))
            if target is None or target not in self._nodes:
                raise RuntimeFault("missing_output")
            transitions += 1
            if transitions > MAX_TRANSITIONS:
                raise RuntimeFault("transition_limit")
            trace.append(TraceEvent(node.id, node.type, "transition", handle))
            return target

        def predict_message(message: str) -> Prediction:
            try:
                prediction = self._predict(message)
            except Exception:
                raise RuntimeFault("nlu_error") from None
            # shortcut: synchronous prediction needs a preemptive outer timeout.
            if monotonic() > deadline:
                raise RuntimeFault("turn_timeout")
            if not isinstance(prediction, Prediction):
                raise RuntimeFault("invalid_nlu_result")
            if prediction.version_id != self._version:
                raise RuntimeFault("model_version_mismatch")
            if (
                not isinstance(prediction.fallback_reasons, tuple)
                or any(
                    not isinstance(reason, str) or reason not in FALLBACK_REASONS
                    for reason in prediction.fallback_reasons
                )
                or not isinstance(prediction.entities, tuple)
                or any(not isinstance(entity, EntityValue) for entity in prediction.entities)
            ):
                raise RuntimeFault("invalid_nlu_result")
            return prediction

        def entity_value(
            ask: AskSlotConfig, slot: Slot, prediction: Prediction, message: str
        ) -> str | int | bool | None:
            if ask.entity_id is None:
                return None
            entity = self._entity_by_id[ask.entity_id]
            matches = [item for item in prediction.entities if item.entity_id == entity.id]
            if not matches:
                return None
            if len(matches) != 1:
                raise SlotValueError("ambiguous_entity")
            match = matches[0]
            if (
                type(match.start) is not int
                or type(match.end) is not int
                or not 0 <= match.start < match.end <= len(message)
                or not isinstance(match.text, str)
                or message[match.start : match.end] != match.text
                or match.source != entity.method
            ):
                raise RuntimeFault("invalid_nlu_result")
            value = match.text
            if entity.method == "dictionary":
                normalized = normalize_example(value)
                entries = [
                    entry.value
                    for entry in entity.dictionary
                    if normalized
                    in {normalize_example(term) for term in (entry.value, *entry.synonyms)}
                ]
                if len(entries) != 1:
                    raise RuntimeFault("invalid_nlu_result")
                value = entries[0]
            return parse_slot_value(slot, value, ask)

        try:
            while True:
                if monotonic() > deadline:
                    raise RuntimeFault("turn_timeout")
                node = self._nodes.get(node_id)
                if node is None:
                    raise RuntimeFault("invalid_checkpoint")
                if node.type in {"start", "switch_intent", "say", "fallback", "end"}:
                    allowed = {
                        "start": set(),
                        "switch_intent": set(),
                        "say": {"text"},
                        "fallback": {"text", "resume_node_id", "example"},
                        "end": {"text"},
                    }[node.type]
                    if set(node.config) - allowed:
                        raise RuntimeFault("unsupported_config")
                if node.type == "start":
                    node_id = move(node, "next")
                elif node.type == "say":
                    messages.append(render_template(str(node.config["text"]), self._slots, values))
                    node_id = move(node, "next")
                elif node.type == "switch_intent":
                    if text is None:
                        trace.append(TraceEvent(node.id, node.type, "wait"))
                        return RuntimeResult(state("waiting_input"), tuple(messages), tuple(trace))
                    prediction = predict_message(text)
                    if prediction.intent_id is None:
                        if not prediction.fallback_reasons:
                            raise RuntimeFault("invalid_nlu_result")
                        trace.append(
                            TraceEvent(
                                node.id,
                                node.type,
                                "fallback",
                                fallback_reasons=prediction.fallback_reasons,
                            )
                        )
                        handle = "fallback"
                    else:
                        if (
                            not isinstance(prediction.intent_id, UUID)
                            or prediction.fallback_reasons
                            or prediction.intent_id not in self._enabled
                        ):
                            raise RuntimeFault("invalid_nlu_result")
                        handle = str(prediction.intent_id)
                        fallback_count = 0
                    text = None
                    last_prediction = prediction
                    node_id = move(node, handle)
                elif node.type == "ask_slot":
                    ask = self._asks[node.id]
                    slot = self._slot_by_id[ask.slot_id]
                    if valid_value(slot, values.get(slot.code), ask):
                        if pending_ask == node.id:
                            text = None
                        pending_ask = None
                        attempts.pop(node.id, None)
                        node_id = move(node, "success")
                        continue
                    values[slot.code] = None
                    if text is not None:
                        answer = text
                        text = None
                        pending_ask = None
                        try:
                            extracted = (
                                entity_value(ask, slot, predict_message(answer), answer)
                                if ask.entity_id is not None
                                else None
                            )
                            value = (
                                extracted
                                if extracted is not None
                                else parse_slot_value(slot, answer, ask)
                            )
                        except SlotValueError:
                            count = attempts.get(node.id, 0) + 1
                            if count >= ask.max_attempts:
                                attempts.pop(node.id, None)
                                node_id = move(node, "exhausted")
                            else:
                                attempts[node.id] = count
                                node_id = move(node, "invalid")
                            continue
                        values[slot.code] = value
                        fallback_count = 0
                        attempts.pop(node.id, None)
                        trace.append(TraceEvent(node.id, node.type, "slot_filled", str(slot.id)))
                        node_id = move(node, "success")
                        continue
                    if (
                        last_prediction is not None
                        and original_text is not None
                        and not entity_consumed
                    ):
                        try:
                            extracted = entity_value(ask, slot, last_prediction, original_text)
                        except SlotValueError:
                            extracted = None
                        if extracted is not None:
                            values[slot.code] = extracted
                            entity_consumed = True
                            fallback_count = 0
                            attempts.pop(node.id, None)
                            trace.append(
                                TraceEvent(node.id, node.type, "slot_filled", str(slot.id))
                            )
                            node_id = move(node, "success")
                            continue
                    messages.append(render_template(ask.prompt, self._slots, values))
                    pending_ask = node.id
                    trace.append(TraceEvent(node.id, node.type, "wait"))
                    return RuntimeResult(state("waiting_input"), tuple(messages), tuple(trace))
                elif node.type == "fallback":
                    messages.append(render_template(str(node.config["text"]), self._slots, values))
                    fallback_count += 1
                    if fallback_count >= 3:
                        example = node.config.get("example")
                        if isinstance(example, str) and example.strip():
                            messages.append(f"Например: {example.strip()}")
                        messages.append("Начните диалог заново и попробуйте другой запрос.")
                        trace.append(TraceEvent(node.id, node.type, "fallback_limit"))
                        values.clear()
                        attempts.clear()
                        pending_ask = None
                        return RuntimeResult(state("completed"), tuple(messages), tuple(trace))
                    raw = node.config.get("resume_node_id")
                    try:
                        resume = UUID(raw) if isinstance(raw, str) else None
                    except ValueError:
                        resume = None
                    if resume not in self._nodes:
                        raise RuntimeFault("invalid_resume")
                    transitions += 1
                    if transitions > MAX_TRANSITIONS:
                        raise RuntimeFault("transition_limit")
                    trace.append(TraceEvent(node.id, node.type, "wait"))
                    assert resume is not None
                    node_id = resume
                    return RuntimeResult(state("waiting_input"), tuple(messages), tuple(trace))
                elif node.type == "end":
                    text_value = node.config.get("text")
                    if isinstance(text_value, str) and text_value:
                        messages.append(render_template(text_value, self._slots, values))
                    trace.append(TraceEvent(node.id, node.type, "complete"))
                    values.clear()
                    attempts.clear()
                    pending_ask = None
                    return RuntimeResult(state("completed"), tuple(messages), tuple(trace))
                else:
                    raise RuntimeFault("unsupported_in_stage")
        except TemplateError as exc:
            return fail(exc.code)
        except RuntimeFault as exc:
            return fail(exc.code)
