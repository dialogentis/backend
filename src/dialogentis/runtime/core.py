"""Deterministic in-memory runtime; callers own persistence, isolation and timeouts."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

from dialogentis.ask_slot import AskSlotConfig
from dialogentis.conditions import ConditionConfig, choose_branch
from dialogentis.datasets.preview import preview_entity
from dialogentis.datasets.schema import Entity, Example, Intent
from dialogentis.datasets.validation import normalize_example
from dialogentis.flow_schema import DEFAULT_FALLBACK_TEXT, Flow, FlowNode
from dialogentis.flow_validation import validate_flow
from dialogentis.handoff import MAX_HANDOFF_DEPTH, HandoffConfig, ReturnConfig, check_mapping
from dialogentis.http_action import (
    HTTP_OUTPUTS,
    HttpActionConfig,
    MockIntegration,
    mock_http_action,
)
from dialogentis.nlu.dialogue import (
    DialogueChoice,
    DialogueDecision,
    DialogueField,
    DialogueTurn,
    DialogueValue,
    accepts_decision,
)
from dialogentis.runtime.history_context import HistoryContext
from dialogentis.runtime.slot_values import SlotValueError, parse_slot_value, valid_value
from dialogentis.set_slot import SetSlotConfig
from dialogentis.slots import Slot, validate_slots
from dialogentis.templates import TemplateError, render_template

MAX_TRANSITIONS = 50
MAX_SECONDS = 20
SAFE_FAILURE = "Не удалось выполнить сценарий. Попробуйте позже."
MASKED = "[скрыто]"


class RuntimeFault(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class FrameState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    node_id: UUID
    fallback_count: int = Field(ge=0, le=3)
    pending_ask: UUID | None = None
    routing_node: UUID | None = None
    slot_values: tuple[tuple[str, StrictBool | StrictInt | StrictStr | None], ...] = Field(
        default=(), repr=False
    )
    attempts: tuple[tuple[UUID, StrictInt], ...] = Field(default=(), repr=False)


class SuspendedFrame(FrameState):
    path_index: int = Field(ge=0, lt=MAX_HANDOFF_DEPTH)


class RuntimeState(FrameState):
    version_id: UUID
    flow_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["waiting_input", "completed", "failed"]
    turn: int = Field(ge=0)
    active_path: tuple[UUID, ...] = Field(default=(), max_length=MAX_HANDOFF_DEPTH)
    history_sources: tuple[UUID | None, ...] = Field(default=(), max_length=MAX_HANDOFF_DEPTH)
    callers: tuple[SuspendedFrame, ...] = Field(
        default=(), max_length=MAX_HANDOFF_DEPTH, repr=False
    )


@dataclass(frozen=True)
class TraceEvent:
    node_id: UUID
    node_type: str
    event: str
    output: str | None = None
    fallback_reasons: tuple[str, ...] = ()
    version_id: UUID | None = None


class NLUObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: UUID
    intent_id: UUID | None = None


@dataclass(frozen=True)
class RuntimeResult:
    state: RuntimeState
    messages: tuple[str, ...]
    trace: tuple[TraceEvent, ...]
    error_code: str | None = None
    history_input: str | None = field(default=None, repr=False)
    history_messages: tuple[str, ...] | None = field(default=None, repr=False)
    nlu_observations: tuple[NLUObservation, ...] = ()


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
    def __init__(
        self,
        version_id: UUID,
        config: dict,
        understand: Callable[[DialogueTurn], DialogueDecision],
        *,
        mock_integrations: list[MockIntegration] | None = None,
        http_handler: Callable[[UUID, dict], tuple[str, dict]] | None = None,
        resolve_version: Callable[[UUID], RuntimeEngine | None] | None = None,
        load_history: Callable[[UUID, UUID], HistoryContext] | None = None,
    ):
        errors, _ = validate_flow(config)
        if errors:
            raise RuntimeFault("invalid_flow")
        try:
            self._flow = Flow.model_validate(config["flow"]).model_copy(deep=True)
            self._slots, slot_errors = validate_slots(config.get("slots", []))
            self._examples = [
                Example.model_validate(item)
                for item in config.get("examples", [])
                if item.get("split") == "train"
            ]
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
        self._slot_by_code = {slot.code: slot for slot in self._slots}
        self._slot_by_id = {slot.id: slot for slot in self._slots}
        self._entity_by_id = {entity.id: entity for entity in self._entities}
        self._asks = {
            node.id: AskSlotConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "ask_slot"
        }
        self._sets = {
            node.id: SetSlotConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "set_slot"
        }
        self._http_actions = {
            node.id: HttpActionConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "http_action"
        }
        self._mock_integrations = (
            {item.id: item for item in mock_integrations} if mock_integrations is not None else None
        )
        self._http_handler = http_handler
        self._resolve_version = resolve_version
        self._load_history = load_history
        self._handoffs = {
            node.id: HandoffConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "handoff"
        }
        self._conditions = {
            node.id: ConditionConfig.model_validate(node.config)
            for node in self._flow.nodes
            if node.type == "condition"
        }
        self._start = next(node.id for node in self._flow.nodes if node.type == "start")
        try:
            self._hash = semantic_hash(self._flow, self._slots, self._intents, self._entities)
        except TypeError, ValueError, UnicodeError:
            raise RuntimeFault("invalid_flow") from None
        self._version = version_id
        self._understand = understand

    def initial_state(self, values: dict | None = None) -> RuntimeState:
        if values is not None and not isinstance(values, dict):
            raise RuntimeFault("invalid_slot_values")
        initial = {slot.code: slot.default for slot in self._slots if slot.default is not None}
        initial.update(values or {})
        self._check_values(initial, "invalid_slot_values")
        return RuntimeState(
            version_id=self._version,
            flow_hash=self._hash,
            node_id=self._start,
            status="waiting_input",
            turn=0,
            fallback_count=0,
            slot_values=tuple(sorted(cast(dict[str, str | int | bool | None], initial).items())),
        )

    def start(self, values: dict | None = None) -> RuntimeResult:
        initial = self.initial_state(values)
        return self._run(self._start, None, 0, 0, dict(initial.slot_values), {}, None)

    def send(self, state: RuntimeState, text: str, values: dict | None = None) -> RuntimeResult:
        if values is not None and not isinstance(values, dict):
            raise RuntimeFault("invalid_slot_values")
        if state.version_id != self._version or state.flow_hash != self._hash:
            raise RuntimeFault("version_mismatch")
        if state.status != "waiting_input":
            raise RuntimeFault("conversation_finished")
        path = [self, *(self._target(version) for version in state.active_path)]
        if len({engine._version for engine in path}) != len(path):  # noqa: SLF001
            raise RuntimeFault("invalid_checkpoint")
        history_sources = state.history_sources or (None,) * len(state.active_path)
        if len(history_sources) != len(state.active_path):
            raise RuntimeFault("invalid_checkpoint")
        for index, source in enumerate(history_sources):
            if source is None:
                continue
            handoff = path[index]._handoffs.get(source)  # noqa: SLF001
            if (
                handoff is None
                or not handoff.include_history
                or handoff.target_version_id != state.active_path[index]
            ):
                raise RuntimeFault("invalid_checkpoint")
        indices = [frame.path_index for frame in state.callers]
        if indices != sorted(set(indices)) or any(index >= len(path) - 1 for index in indices):
            raise RuntimeFault("invalid_checkpoint")
        for frame in state.callers:
            parent = path[frame.path_index]
            parent._check_frame(frame)  # noqa: SLF001
            call = parent._handoffs.get(frame.node_id)  # noqa: SLF001
            if (
                call is None
                or call.mode != "call"
                or call.target_version_id != path[frame.path_index + 1]._version  # noqa: SLF001
            ):
                raise RuntimeFault("invalid_checkpoint")
        current = path[-1]
        current._check_frame(state)  # noqa: SLF001
        if not isinstance(text, str) or not 1 <= len(text) <= 2000:
            raise RuntimeFault("invalid_message")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise RuntimeFault("invalid_message") from None
        stored = dict(state.slot_values)
        stored.update(values or {})
        current._check_values(stored, "invalid_slot_values")  # noqa: SLF001
        return self._run(
            state.node_id,
            text,
            state.turn + 1,
            state.fallback_count,
            stored,
            dict(state.attempts),
            state.pending_ask,
            path,
            list(state.callers),
            list(history_sources),
            state.routing_node,
        )

    def _target(self, version_id: UUID) -> RuntimeEngine:
        engine = self._resolve_version(version_id) if self._resolve_version is not None else None
        if engine is None or engine._version != version_id:
            raise RuntimeFault("handoff_target_unavailable")
        return engine

    def _check_frame(self, state: FrameState) -> None:
        if state.node_id not in self._nodes or state.fallback_count >= 3:
            raise RuntimeFault("invalid_checkpoint")
        if state.pending_ask is not None and (
            state.pending_ask != state.node_id or state.pending_ask not in self._asks
        ):
            raise RuntimeFault("invalid_checkpoint")
        if state.routing_node is not None and (
            (router := self._nodes.get(state.routing_node)) is None
            or router.type != "switch_intent"
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
        path: list[RuntimeEngine] | None = None,
        callers: list[SuspendedFrame] | None = None,
        history_sources: list[UUID | None] | None = None,
        routing_node: UUID | None = None,
    ) -> RuntimeResult:
        messages: list[str] = []
        trace: list[TraceEvent] = []
        nlu_observations: list[NLUObservation] = []
        transitions = 0
        http_actions = 0
        deadline = monotonic() + MAX_SECONDS
        original_text = text
        dialogue_values: dict[str, DialogueValue] = {}
        fallback_reply: str | None = None
        path = path if path is not None else [self]
        callers = callers if callers is not None else []
        history_sources = history_sources if history_sources is not None else []
        active = path[-1]
        active_history: HistoryContext | None = None
        input_consumed = False
        private_input = False
        private_values: set[str] = set()

        def collect(engine: RuntimeEngine, slots: dict) -> None:
            for code, value in slots.items():
                if value is not None and engine._slot_by_code[code].sensitive:
                    if type(value) is bool:
                        private_values.update(("true", "да") if value else ("false", "нет"))
                    elif value != "":
                        private_values.add(str(value))

        for caller in callers:
            collect(path[caller.path_index], dict(caller.slot_values))

        def record(event: TraceEvent) -> None:
            trace.append(replace(event, version_id=active._version))

        def redactor() -> Callable[[str], str]:
            tokens = sorted(private_values, key=lambda value: (-len(value), value))
            if len(tokens) > 200 or sum(map(len, tokens)) > 40_000:
                return lambda _: MASKED
            if not tokens:
                return lambda value: value[:4000]
            pattern = re.compile("|".join(map(re.escape, tokens)), re.IGNORECASE)
            return lambda value: pattern.sub(MASKED, value)[:4000]

        def result(status: Literal["waiting_input", "completed", "failed"], code=None):
            redact = redactor()
            return RuntimeResult(
                state(status),
                tuple(messages),
                tuple(trace),
                code,
                None
                if turn == 0 or initial_input is None or status == "failed"
                else (MASKED if private_input else redact(initial_input)),
                tuple(redact(message) for message in messages),
                tuple(nlu_observations),
            )

        initial_input = text

        def state(status: Literal["waiting_input", "completed", "failed"]) -> RuntimeState:
            return RuntimeState(
                version_id=self._version,
                flow_hash=self._hash,
                node_id=node_id,
                status=status,
                turn=turn,
                fallback_count=fallback_count,
                pending_ask=pending_ask,
                routing_node=routing_node if status == "waiting_input" else None,
                slot_values=tuple(sorted(values.items())),
                attempts=tuple((node, attempts[node]) for node in sorted(attempts, key=str)),
                active_path=tuple(engine._version for engine in path[1:])
                if status == "waiting_input"
                else (),
                history_sources=tuple(history_sources)
                if status == "waiting_input" and any(history_sources)
                else (),
                callers=tuple(callers) if status == "waiting_input" else (),
            )

        def fail(code: str) -> RuntimeResult:
            nonlocal pending_ask
            failed_node = active._nodes.get(node_id)
            record(
                TraceEvent(node_id, failed_node.type if failed_node else "unknown", "error", code)
            )
            values.clear()
            attempts.clear()
            pending_ask = None
            messages.append(SAFE_FAILURE)
            return result("failed", code)

        def move(node: FlowNode, handle: str) -> UUID:
            nonlocal transitions
            target = active._edges.get((node.id, handle))
            if target is None or target not in active._nodes:
                raise RuntimeFault("missing_output")
            transitions += 1
            if transitions > MAX_TRANSITIONS:
                raise RuntimeFault("transition_limit")
            record(TraceEvent(node.id, node.type, "transition", handle))
            return target

        def dialogue_history() -> HistoryContext:
            nonlocal active_history
            if active_history is None:
                active_history = (
                    self._load_history(path[-2]._version, source)
                    if history_sources and (source := history_sources[-1]) and self._load_history
                    else HistoryContext()
                )
            return active_history

        def understand(
            message: str, node: FlowNode, router: UUID | None, ask_slot: Slot | None = None
        ) -> DialogueDecision:
            examples: dict[UUID, list[str]] = {}
            for example in active._examples:
                if example.intent_id is None:
                    continue
                phrases = examples.setdefault(example.intent_id, [])
                if len(phrases) < 2:
                    phrases.append(example.text[:240])
            choices = tuple(
                DialogueChoice(
                    code=intent.code,
                    name=intent.name,
                    description=intent.description[:500],
                    examples=tuple(examples.get(intent.id, ())),
                )
                for intent in active._intents
                if intent.enabled
                and router is not None
                and (router, str(intent.id)) in active._edges
            )
            fields: dict[str, DialogueField] = {}
            asks = list(active._asks.values())
            if ask_slot is not None:
                asks.insert(0, active._asks[node.id])
            for ask in asks:
                slot = active._slot_by_id[ask.slot_id]
                if slot.sensitive or slot.value_type == "boolean" or slot.code in fields:
                    continue
                fields[slot.code] = DialogueField(
                    code=slot.code,
                    name=slot.name,
                    value_type=slot.value_type,
                    prompt=ask.prompt[:500],
                    entity=active._entity_by_id[ask.entity_id].name if ask.entity_id else None,
                )
            safe_text = redactor()(message)[:2000]
            fallback = active._nodes.get(active._edges.get((node.id, "fallback"), node.id))
            guidance = (
                fallback.config.get("reply_instruction")
                if node.type == "switch_intent" and fallback and fallback.type == "fallback"
                else None
            )
            request = DialogueTurn(
                version_id=active._version,
                node_id=node.id,
                text=safe_text,
                input_hash=hashlib.sha256(message.encode()).hexdigest()
                if safe_text != message
                else None,
                history=dialogue_history(),
                choices=choices,
                fields=tuple(fields.values())[:32],
                awaiting=ask_slot.code if ask_slot is not None else None,
                fallback_instruction=guidance if isinstance(guidance, str) else None,
            )
            try:
                decision = active._understand(request)
                if not accepts_decision(request, decision):
                    raise ValueError("Invalid dialogue decision")
            except Exception:
                raise RuntimeFault("nlu_error") from None
            if monotonic() > deadline:
                raise RuntimeFault("turn_timeout")
            return decision

        def route_decision(decision: DialogueDecision) -> UUID | None:
            intent_id = next(
                (intent.id for intent in active._intents if intent.code == decision.intent), None
            )
            nlu_observations.append(NLUObservation(version_id=active._version, intent_id=intent_id))
            return intent_id

        def parse_proposal(
            slot: Slot, ask: AskSlotConfig, value: DialogueValue
        ) -> str | int | bool:
            entity = active._entity_by_id.get(ask.entity_id) if ask.entity_id else None
            if entity is None or entity.method == "learned":
                raw = value.source if slot.value_type == "string" else value.value
            elif entity.method == "dictionary":
                normalized = normalize_example(value.source)
                raw = next(
                    (
                        entry.value
                        for entry in entity.dictionary
                        if normalized
                        in {normalize_example(term) for term in (entry.value, *entry.synonyms)}
                    ),
                    "",
                )
            else:
                matches = preview_entity(entity, original_text or "").matches
                if not any(match.text == value.source for match in matches):
                    raise SlotValueError()
                raw = value.source
            return parse_slot_value(slot, raw, ask)

        try:
            while True:
                collect(active, values)
                if monotonic() > deadline:
                    raise RuntimeFault("turn_timeout")
                node = active._nodes.get(node_id)
                if node is None:
                    raise RuntimeFault("invalid_checkpoint")
                if node.type in {"start", "switch_intent", "say", "fallback", "end"}:
                    allowed = {
                        "start": set(),
                        "switch_intent": set(),
                        "say": {"text"},
                        "fallback": {"text", "resume_node_id", "example", "reply_instruction"},
                        "end": {"text"},
                    }[node.type]
                    if set(node.config) - allowed:
                        raise RuntimeFault("unsupported_config")
                if node.type == "start":
                    node_id = move(node, "next")
                elif node.type == "say":
                    messages.append(
                        render_template(str(node.config["text"]), active._slots, values)
                    )
                    node_id = move(node, "next")
                elif node.type == "switch_intent":
                    routing_node = node.id
                    if text is None:
                        record(TraceEvent(node.id, node.type, "wait"))
                        return result("waiting_input")
                    decision = understand(text, node, node.id)
                    intent_id = route_decision(decision)
                    dialogue_values = {value.code: value for value in decision.values}
                    text = None
                    if intent_id is None:
                        fallback_reply = decision.reply
                        record(
                            TraceEvent(
                                node.id,
                                node.type,
                                "fallback",
                                fallback_reasons=("ambiguous_intent",),
                            )
                        )
                        node_id = move(node, "fallback")
                    else:
                        fallback_count = 0
                        node_id = move(node, str(intent_id))
                elif node.type == "condition":
                    handle = choose_branch(active._conditions[node.id], active._slot_by_id, values)
                    node_id = move(node, handle)
                elif node.type == "http_action" and (
                    active._mock_integrations is not None or active._http_handler is not None
                ):
                    http_actions += 1
                    if http_actions > 3:
                        raise RuntimeFault("http_action_limit")
                    action = active._http_actions[node.id]
                    if active._http_handler is not None:
                        codes = {
                            active._slot_by_id[slot_id].code
                            for mapping in (action.query, action.body)
                            for slot_id in mapping.values()
                        }
                        outcome, updates = active._http_handler(
                            node.id, {code: values.get(code) for code in codes}
                        )
                        allowed = {
                            active._slot_by_id[slot_id].code
                            for slot_id in action.response_mapping.values()
                        }
                        if (
                            outcome not in HTTP_OUTPUTS
                            or not isinstance(updates, dict)
                            or (set(updates) - allowed or (outcome != "success" and updates))
                        ):
                            raise RuntimeFault("invalid_http_result")
                        active._check_values(updates, "invalid_http_result")
                        values.update(updates)
                    else:
                        assert active._mock_integrations is not None
                        outcome = mock_http_action(
                            action,
                            active._mock_integrations.get(action.integration_id),
                            active._slot_by_id,
                            values,
                        )
                    node_id = move(node, outcome)
                elif node.type == "set_slot":
                    assignment = active._sets[node.id]
                    slot = active._slot_by_id[assignment.slot_id]
                    if assignment.operation == "clear":
                        values[slot.code] = None
                        node_id = move(node, "next")
                    else:
                        source = active._slot_by_id.get(assignment.source_slot_id)
                        value = assignment.value
                        if assignment.source == "slot":
                            value = None
                            if source and (slot.sensitive or not source.sensitive):
                                value = values.get(source.code)
                        if value is None:
                            node_id = move(node, "error")
                        else:
                            values[slot.code] = cast(str | int | bool, value)
                            record(TraceEvent(node.id, node.type, "slot_set", str(slot.id)))
                            node_id = move(node, "next")
                elif node.type == "ask_slot":
                    ask = active._asks[node.id]
                    slot = active._slot_by_id[ask.slot_id]
                    if slot.sensitive and text is not None:
                        private_input = True
                    if slot.code in dialogue_values:
                        try:
                            value = parse_proposal(slot, ask, dialogue_values.pop(slot.code))
                        except SlotValueError:
                            pass
                        else:
                            values[slot.code] = value
                            fallback_count = 0
                            attempts.pop(node.id, None)
                            record(TraceEvent(node.id, node.type, "slot_filled", str(slot.id)))
                            node_id = move(node, "success")
                            continue
                    if valid_value(slot, values.get(slot.code), ask):
                        if pending_ask == node.id:
                            text = None
                            input_consumed = True
                        pending_ask = None
                        attempts.pop(node.id, None)
                        node_id = move(node, "success")
                        continue
                    values[slot.code] = None
                    if text is not None:
                        answer = text
                        text = None
                        input_consumed = True
                        pending_ask = None
                        try:
                            if slot.sensitive or slot.value_type == "boolean":
                                value = parse_slot_value(slot, answer, ask)
                            else:
                                decision = understand(answer, node, routing_node, slot)
                                understood = {value.code: value for value in decision.values}
                                if decision.action == "route" and routing_node is not None:
                                    intent_id = route_decision(decision)
                                    attempts.clear()
                                    dialogue_values = understood
                                    fallback_count = 0
                                    node_id = move(active._nodes[routing_node], str(intent_id))
                                    continue
                                if decision.action != "continue" or slot.code not in understood:
                                    raise SlotValueError()
                                value = parse_proposal(slot, ask, understood.pop(slot.code))
                                dialogue_values = understood
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
                        record(TraceEvent(node.id, node.type, "slot_filled", str(slot.id)))
                        node_id = move(node, "success")
                        continue
                    messages.append(render_template(ask.prompt, active._slots, values))
                    pending_ask = node.id
                    record(TraceEvent(node.id, node.type, "wait"))
                    return result("waiting_input")
                elif node.type == "fallback":
                    if fallback_reply is not None:
                        messages.append(fallback_reply)
                        fallback_reply = None
                        fallback_count = 0
                        record(TraceEvent(node.id, node.type, "generated_reply"))
                    else:
                        text_template = node.config.get("text")
                        reply = (
                            render_template(text_template, active._slots, values)
                            if isinstance(text_template, str) and text_template.strip()
                            else DEFAULT_FALLBACK_TEXT
                        )
                        messages.append(reply)
                        fallback_count += 1
                    if fallback_count >= 3:
                        example = node.config.get("example")
                        if isinstance(example, str) and example.strip():
                            messages.append(f"Например: {example.strip()}")
                        messages.append("Начните диалог заново и попробуйте другой запрос.")
                        record(TraceEvent(node.id, node.type, "fallback_limit"))
                        values.clear()
                        attempts.clear()
                        pending_ask = None
                        return result("completed")
                    raw = node.config.get("resume_node_id")
                    try:
                        resume = UUID(raw) if isinstance(raw, str) else None
                    except ValueError:
                        resume = None
                    if resume not in active._nodes:
                        raise RuntimeFault("invalid_resume")
                    transitions += 1
                    if transitions > MAX_TRANSITIONS:
                        raise RuntimeFault("transition_limit")
                    record(TraceEvent(node.id, node.type, "wait"))
                    assert resume is not None
                    node_id = resume
                    return result("waiting_input")
                elif node.type == "handoff":
                    config = active._handoffs[node.id]
                    if len(path) > MAX_HANDOFF_DEPTH:
                        raise RuntimeFault("handoff_depth")
                    if config.target_version_id in {engine._version for engine in path}:
                        raise RuntimeFault("handoff_cycle")
                    try:
                        target = self._target(config.target_version_id)
                        check_mapping(config.input_mapping, active._slot_by_id, target._slot_by_id)
                        check_mapping(config.return_mapping, target._slot_by_id, active._slot_by_id)
                        initial = target.initial_state(
                            _map_slots(
                                config.input_mapping, active._slot_by_id, target._slot_by_id, values
                            )
                        )
                    except RuntimeFault, ValueError:
                        node_id = move(node, "error")
                        continue
                    transitions += 1
                    if transitions > MAX_TRANSITIONS:
                        raise RuntimeFault("transition_limit")
                    record(TraceEvent(node.id, node.type, "handoff", str(target._version)))
                    history_sources.append(node.id if config.include_history else None)
                    active_history = HistoryContext()
                    if config.include_history:
                        if turn > 0 and self._load_history is not None:
                            active_history = self._load_history(active._version, node.id)
                        event = (
                            "handoff_history_truncated"
                            if active_history.truncated
                            else "handoff_history"
                        )
                        record(
                            TraceEvent(node.id, node.type, event, str(len(active_history.messages)))
                        )
                    if config.mode == "call":
                        callers.append(
                            SuspendedFrame(
                                path_index=len(path) - 1,
                                node_id=node.id,
                                fallback_count=fallback_count,
                                pending_ask=pending_ask,
                                routing_node=routing_node,
                                slot_values=tuple(sorted(values.items())),
                                attempts=tuple(attempts.items()),
                            )
                        )
                    path.append(target)
                    active = target
                    node_id, values, attempts = initial.node_id, dict(initial.slot_values), {}
                    fallback_count, pending_ask = 0, None
                    routing_node = None
                    dialogue_values.clear()
                    text = (
                        original_text
                        if config.forward_current_message and not input_consumed
                        else None
                    )
                    original_text = text
                elif node.type == "return" and callers:
                    caller = callers.pop()
                    parent = path[caller.path_index]
                    call = parent._handoffs[caller.node_id]
                    output = ReturnConfig.model_validate(node.config)
                    exported = {
                        source: destination
                        for source, destination in call.return_mapping.items()
                        if source in output.slot_ids
                    }
                    restored = dict(caller.slot_values)
                    branch = "returned"
                    try:
                        check_mapping(exported, active._slot_by_id, parent._slot_by_id)
                        restored.update(
                            _map_slots(exported, active._slot_by_id, parent._slot_by_id, values)
                        )
                    except ValueError:
                        branch = "error"
                    record(TraceEvent(node.id, node.type, "returned", str(parent._version)))
                    del path[caller.path_index + 1 :]
                    del history_sources[caller.path_index :]
                    active = parent
                    active_history = None
                    node_id = caller.node_id
                    values, attempts = restored, dict(caller.attempts)
                    fallback_count, pending_ask = caller.fallback_count, caller.pending_ask
                    routing_node = caller.routing_node
                    dialogue_values.clear()
                    text = original_text = None
                    node_id = move(active._nodes[node_id], branch)
                elif node.type in {"end", "return"}:
                    text_value = node.config.get("text")
                    if isinstance(text_value, str) and text_value:
                        messages.append(render_template(text_value, active._slots, values))
                    record(TraceEvent(node.id, node.type, "complete"))
                    values.clear()
                    attempts.clear()
                    pending_ask = None
                    return result("completed")
                else:
                    raise RuntimeFault("unsupported_in_stage")
        except TemplateError as exc:
            return fail(exc.code)
        except RuntimeFault as exc:
            return fail(exc.code)


def _map_slots(
    mapping: dict[UUID, UUID], sources: dict[UUID, Slot], targets: dict[UUID, Slot], values: dict
) -> dict:
    return {
        targets[target].code: values[sources[source].code]
        for source, target in mapping.items()
        if sources[source].code in values
    }
