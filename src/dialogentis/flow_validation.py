"""Structural checks that do not depend on the still-incomplete runtime schema."""

from collections import defaultdict
from uuid import UUID

from pydantic import ValidationError

from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow


def validate_flow(config: dict) -> tuple[list[Issue], list[Issue]]:
    try:
        flow = Flow.model_validate(config.get("flow"))
    except ValidationError:
        return [Issue(("flow",), "invalid_flow", "Некорректная структура сценария")], []

    errors: list[Issue] = []
    warnings: list[Issue] = []
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
        }
        if source_node.type in allowed and edge.source_handle not in allowed[source_node.type]:
            errors.append(
                Issue((*path, "source_handle"), "invalid_output", "У этого блока нет такого выхода")
            )
        adjacency[edge.source].append(edge.target)
        if source_node.type in {"start", "say", "set_slot", "condition", "http_action"} or (
            source_node.type == "ask_slot" and edge.source_handle == "success"
        ):
            immediate[edge.source].append(edge.target)

    for index, node in enumerate(flow.nodes):
        path = ("flow", "nodes", index)
        required = {"start": ("next",), "say": ("next",), "switch_intent": ("fallback",)}
        for handle in required.get(node.type, ()):
            if (node.id, handle) not in handles:
                errors.append(
                    Issue(path, "missing_output", "Не подключён обязательный выход блока")
                )
        if node.type == "fallback":
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
