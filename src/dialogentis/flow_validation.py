"""Structural checks that do not depend on the still-incomplete runtime schema."""

from collections import defaultdict
from uuid import UUID

from pydantic import ValidationError

from dialogentis.ask_slot import AskSlotConfig, check_ask_config
from dialogentis.conditions import ConditionConfig, validate_condition
from dialogentis.datasets.schema import Entity, Intent
from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.handoff import handoff_outputs, validate_handoff_references
from dialogentis.http_action import HTTP_OUTPUTS, HttpActionConfig, check_http_references
from dialogentis.set_slot import SetSlotConfig, check_set_config
from dialogentis.slots import validate_slots
from dialogentis.templates import TemplateError, parse_template


def validate_intent_outputs(flow: Flow, intents: list[Intent]) -> tuple[list[Issue], list[Issue]]:
    intent_by_id = {str(intent.id): intent for intent in intents}
    switches = {node.id for node in flow.nodes if node.type == "switch_intent"}
    errors, warnings = [], []
    for index, edge in enumerate(flow.edges):
        if edge.source not in switches or edge.source_handle == "fallback":
            continue
        intent = intent_by_id.get(edge.source_handle)
        path = ("flow", "edges", index, "source_handle")
        if intent is None:
            errors.append(
                Issue(path, "unknown_intent_output", "Ветвь ссылается на отсутствующее намерение")
            )
        elif not intent.enabled:
            warnings.append(
                Issue(
                    path,
                    "disabled_intent_output",
                    "Намерение ветви отключено и не будет распознано",
                )
            )
    return errors, warnings


def validate_flow(config: dict) -> tuple[list[Issue], list[Issue]]:
    try:
        flow = Flow.model_validate(config.get("flow"))
    except ValidationError:
        return [Issue(("flow",), "invalid_flow", "Некорректная структура сценария")], []

    errors: list[Issue] = []
    warnings: list[Issue] = []
    slots, slot_errors = validate_slots(config.get("slots", []))
    errors.extend(slot_errors)
    errors.extend(validate_handoff_references(flow, slots, complete=True))
    slot_by_id = {slot.id: slot for slot in slots}
    entity_by_id = {}
    if any(node.type == "ask_slot" for node in flow.nodes):
        try:
            entities = config.get("entities", [])
            if not isinstance(entities, list):
                raise ValueError("Invalid entities")
            parsed_entities = [Entity.model_validate(item) for item in entities]
            entity_by_id = {entity.id: entity for entity in parsed_entities}
            if len(entity_by_id) != len(parsed_entities):
                raise ValueError("Duplicate entities")
        except ValidationError, ValueError:
            errors.append(Issue(("entities",), "invalid_entities", "Проверьте данные сущностей"))
    try:
        intents = config.get("intents", [])
        if not isinstance(intents, list):
            raise ValueError("Invalid intents")
        branch_errors, branch_warnings = validate_intent_outputs(
            flow, [Intent.model_validate(item) for item in intents]
        )
        errors.extend(branch_errors)
        warnings.extend(branch_warnings)
    except ValidationError, ValueError:
        errors.append(Issue(("intents",), "invalid_intents", "Проверьте данные намерений"))
    nodes = {}
    for index, node in enumerate(flow.nodes):
        if node.id in nodes:
            errors.append(
                Issue(
                    ("flow", "nodes", index, "id"),
                    "duplicate_node",
                    "Идентификатор блока повторяется",
                    nodes[node.id],
                )
            )
        else:
            nodes[node.id] = index
    starts = [index for index, node in enumerate(flow.nodes) if node.type == "start"]
    if len(starts) != 1:
        errors.append(
            Issue(("flow", "nodes"), "start_count", "В сценарии должен быть ровно один блок начала")
        )

    adjacency: dict[UUID, list[UUID]] = defaultdict(list)
    immediate: dict[UUID, list[UUID]] = defaultdict(list)
    edge_ids: dict[UUID, int] = {}
    handles: dict[tuple[UUID, str], int] = {}
    for index, edge in enumerate(flow.edges):
        path = ("flow", "edges", index)
        if edge.id in edge_ids:
            errors.append(
                Issue(
                    (*path, "id"),
                    "duplicate_edge",
                    "Идентификатор связи повторяется",
                    edge_ids[edge.id],
                )
            )
        else:
            edge_ids[edge.id] = index
        handle = edge.source, edge.source_handle
        if handle in handles:
            errors.append(
                Issue(
                    (*path, "source_handle"),
                    "duplicate_handle",
                    "Из одного выхода допускается только одна связь",
                    handles[handle],
                )
            )
        else:
            handles[handle] = index
        source, target = nodes.get(edge.source), nodes.get(edge.target)
        if source is None:
            errors.append(
                Issue((*path, "source"), "missing_source", "Начальный блок связи не найден")
            )
        if target is None:
            errors.append(
                Issue((*path, "target"), "missing_target", "Конечный блок связи не найден")
            )
        if source is None or target is None:
            continue
        source_node, target_node = flow.nodes[source], flow.nodes[target]
        if target_node.type == "start":
            errors.append(
                Issue(
                    (*path, "target"),
                    "start_has_input",
                    "У блока начала не может быть входящих связей",
                )
            )
        allowed = {
            "start": {"next"},
            "say": {"next"},
            "end": set(),
            "return": set(),
            "fallback": set(),
            "ask_slot": {"success", "invalid", "exhausted"},
            "set_slot": {"next", "error"},
            "http_action": set(HTTP_OUTPUTS),
            "handoff": set(handoff_outputs(str(source_node.config.get("mode", "call")))),
        }
        if source_node.type == "condition":
            try:
                condition = ConditionConfig.model_validate(source_node.config)
                allowed["condition"] = {"else", *(str(branch.id) for branch in condition.branches)}
            except ValidationError:
                pass
        if source_node.type in allowed and edge.source_handle not in allowed[source_node.type]:
            errors.append(
                Issue((*path, "source_handle"), "invalid_output", "У этого блока нет такого выхода")
            )
        adjacency[edge.source].append(edge.target)
        if source_node.type in {
            "start",
            "say",
            "set_slot",
            "condition",
            "http_action",
            "handoff",
        } or (source_node.type == "ask_slot" and edge.source_handle == "success"):
            immediate[edge.source].append(edge.target)

    for index, node in enumerate(flow.nodes):
        path = ("flow", "nodes", index)
        if node.type in {"say", "fallback", "end"}:
            text = node.config.get("text")
            if node.type != "end" and (not isinstance(text, str) or not text.strip()):
                errors.append(
                    Issue((*path, "config", "text"), "missing_text", "Укажите текст ответа")
                )
            elif text is not None:
                if not isinstance(text, str):
                    errors.append(
                        Issue(
                            (*path, "config", "text"),
                            "invalid_text",
                            "Текст ответа должен быть строкой",
                        )
                    )
                elif not slot_errors:
                    try:
                        parse_template(text, slots)
                    except TemplateError as exc:
                        message = {
                            "template_too_long": "Текст ответа превышает 4000 символов",
                            "html_not_allowed": "HTML в тексте ответа не поддерживается",
                            "unknown_template_slot": "В тексте указан несуществующий параметр",
                        }.get(exc.code, "Проверьте шаблон текста ответа")
                        errors.append(Issue((*path, "config", "text"), exc.code, message))
        required = {
            "start": ("next",),
            "say": ("next",),
            "switch_intent": ("fallback",),
            "ask_slot": ("success", "invalid", "exhausted"),
            "set_slot": ("next", "error"),
            "http_action": HTTP_OUTPUTS,
            "handoff": handoff_outputs(str(node.config.get("mode", "call")))
            if node.type == "handoff"
            else (),
        }
        for handle in required.get(node.type, ()):
            if (node.id, handle) not in handles:
                errors.append(
                    Issue(path, "missing_output", "Не подключён обязательный выход блока")
                )
        if node.type == "http_action":
            try:
                check_http_references(HttpActionConfig.model_validate(node.config), slot_by_id)
            except ValidationError, ValueError:
                errors.append(
                    Issue(
                        (*path, "config"),
                        "invalid_http_action",
                        "Проверьте настройки HTTP-запроса и параметры",
                    )
                )
        if node.type == "condition":
            try:
                condition = ConditionConfig.model_validate(node.config)
                validate_condition(condition, slot_by_id)
            except ValidationError, ValueError:
                errors.append(
                    Issue((*path, "config"), "invalid_condition", "Проверьте условия и параметры")
                )
            else:
                for handle in ("else", *(str(branch.id) for branch in condition.branches)):
                    if (node.id, handle) not in handles:
                        errors.append(
                            Issue(path, "missing_output", "Не подключён обязательный выход блока")
                        )
        if node.type == "set_slot":
            try:
                check_set_config(SetSlotConfig.model_validate(node.config), slot_by_id)
            except ValidationError, ValueError:
                errors.append(
                    Issue(
                        (*path, "config"),
                        "invalid_set_slot",
                        "Проверьте параметр и значение присваивания",
                    )
                )
        if node.type == "ask_slot":
            try:
                ask = AskSlotConfig.model_validate(node.config)
            except ValidationError:
                errors.append(
                    Issue(
                        (*path, "config"),
                        "invalid_ask_slot",
                        "Настройте параметр, текст вопроса и проверки",
                    )
                )
            else:
                slot = slot_by_id.get(ask.slot_id)
                if slot is None:
                    errors.append(
                        Issue(
                            (*path, "config", "slot_id"),
                            "unknown_slot",
                            "Параметр для сбора не найден",
                        )
                    )
                else:
                    try:
                        check_ask_config(ask, slot, slots)
                    except TemplateError, ValueError:
                        errors.append(
                            Issue(
                                (*path, "config"),
                                "invalid_ask_slot",
                                "Проверьте текст вопроса и проверки параметра",
                            )
                        )
                    if ask.entity_id is not None:
                        entity = entity_by_id.get(ask.entity_id)
                        if entity is None:
                            errors.append(
                                Issue(
                                    (*path, "config", "entity_id"),
                                    "unknown_entity",
                                    "Сущность для сбора не найдена",
                                )
                            )
                        elif entity.value_type != slot.value_type:
                            errors.append(
                                Issue(
                                    (*path, "config", "entity_id"),
                                    "entity_type_mismatch",
                                    "Типы сущности и параметра должны совпадать",
                                )
                            )
        if node.type == "fallback":
            example = node.config.get("example")
            if example is None or (isinstance(example, str) and not example.strip()):
                warnings.append(
                    Issue(
                        (*path, "config", "example"),
                        "missing_fallback_example",
                        "Добавьте пример запроса для повторных ошибок распознавания",
                    )
                )
            elif not isinstance(example, str) or len(example) > 500:
                errors.append(
                    Issue(
                        (*path, "config", "example"),
                        "invalid_fallback_example",
                        "Пример запроса должен быть текстом не длиннее 500 символов",
                    )
                )
            else:
                try:
                    parse_template(example, [])
                except TemplateError:
                    errors.append(
                        Issue(
                            (*path, "config", "example"),
                            "invalid_fallback_example",
                            "Пример запроса должен быть обычным текстом без шаблонов и HTML",
                        )
                    )
            try:
                raw = node.config.get("resume_node_id")
                resume = UUID(raw) if isinstance(raw, str) else None
            except ValueError:
                resume = None
            if resume not in nodes:
                errors.append(
                    Issue(
                        (*path, "config", "resume_node_id"),
                        "invalid_resume",
                        "Выберите существующий блок для продолжения",
                    )
                )
            else:
                assert resume is not None
                if flow.nodes[nodes[resume]].type == "start":
                    errors.append(
                        Issue(
                            (*path, "config", "resume_node_id"),
                            "start_has_input",
                            "У блока начала не может быть входящих связей",
                        )
                    )
                adjacency[node.id].append(resume)

    if len(starts) == 1 and len(nodes) == len(flow.nodes):
        visited: set[UUID] = set()
        pending = [flow.nodes[starts[0]].id]
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            pending.extend(adjacency[node_id])
        for index, node in enumerate(flow.nodes):
            if node.id not in visited:
                warnings.append(
                    Issue(
                        ("flow", "nodes", index),
                        "unreachable_node",
                        "Блок недостижим из начала сценария",
                    )
                )

    if len(nodes) == len(flow.nodes):
        state: dict[UUID, int] = {}

        def visit(node_id: UUID) -> None:
            state[node_id] = 1
            for target in immediate[node_id]:
                if state.get(target) == 1:
                    errors.append(
                        Issue(
                            ("flow", "nodes", nodes[node_id]),
                            "immediate_cycle",
                            "Цикл может выполняться без ожидания нового сообщения",
                        )
                    )
                elif state.get(target, 0) == 0:
                    visit(target)
            state[node_id] = 2

        for node in flow.nodes:
            if state.get(node.id, 0) == 0:
                visit(node.id)

    return errors, warnings
