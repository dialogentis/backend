"""Resolve and validate the ready versions referenced by a draft."""

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.datasets.validation import Issue
from dialogentis.flow_schema import Flow
from dialogentis.flow_validation import validate_flow
from dialogentis.handoff import MAX_HANDOFF_DEPTH, HandoffConfig, check_mapping
from dialogentis.jobs.models import Job
from dialogentis.slots import Slot, validate_slots
from dialogentis.versions.models import AgentVersion


async def validate_handoff_dependencies(
    db: AsyncSession, org_id: UUID, agent_id: UUID, config: dict
) -> list[Issue]:
    errors, _ = await resolve_handoff_dependencies(db, org_id, agent_id, config)
    return errors


async def resolve_handoff_dependencies(
    db: AsyncSession, org_id: UUID, agent_id: UUID, config: dict
) -> tuple[list[Issue], list[AgentVersion]]:
    versions: dict[UUID, AgentVersion | None] = {}
    validated: dict[UUID, tuple[int, frozenset[UUID]]] = {}

    async def load(version_id: UUID):
        if version_id not in versions:
            versions[version_id] = await db.scalar(
                select(AgentVersion)
                .join(Job, (Job.id == AgentVersion.job_id) & (Job.organization_id == org_id))
                .where(
                    AgentVersion.organization_id == org_id,
                    AgentVersion.id == version_id,
                    AgentVersion.model_artifact_id.is_not(None),
                    AgentVersion.validation_report.is_not(None),
                    Job.kind == "agent.training",
                    Job.status == "succeeded",
                )
            )
        return versions[version_id]

    def definitions(payload: dict) -> dict[UUID, Slot]:
        slots, errors = validate_slots(payload.get("slots", []))
        if errors:
            raise ValueError("Invalid parameter definitions")
        return {slot.id: slot for slot in slots}

    async def check(
        call: HandoffConfig, source_slots: dict[UUID, Slot], ancestors: tuple[UUID, ...]
    ) -> tuple[int, frozenset[UUID]]:
        if call.target_agent_id in ancestors:
            raise ValueError("Передача образует цикл между агентами")
        if len(ancestors) > MAX_HANDOFF_DEPTH:
            raise ValueError("Цепочка передач превышает пять уровней")
        version = await load(call.target_version_id)
        if version is None or version.agent_id != call.target_agent_id:
            raise ValueError("Выберите готовую версию агента из этой организации")
        payload = version.snapshot["config"]
        target_slots = definitions(payload)
        try:
            check_mapping(call.input_mapping, source_slots, target_slots)
            check_mapping(call.return_mapping, target_slots, source_slots)
        except ValueError:
            raise ValueError("Проверьте совместимость параметров передачи и возврата") from None
        cached = validated.get(version.id)
        if cached is not None:
            depth, agents = cached
            if agents.intersection(ancestors):
                raise ValueError("Передача образует цикл между агентами")
            if len(ancestors) + depth - 1 > MAX_HANDOFF_DEPTH:
                raise ValueError("Цепочка передач превышает пять уровней")
            return cached
        if validate_flow(payload)[0]:
            raise ValueError("Исправьте ошибки сценария целевой версии")
        depth, agents = 1, {version.agent_id}
        for node in Flow.model_validate(payload["flow"]).nodes:
            if node.type == "handoff":
                child_depth, child_agents = await check(
                    HandoffConfig.model_validate(node.config),
                    target_slots,
                    (*ancestors, version.agent_id),
                )
                depth = max(depth, child_depth + 1)
                agents.update(child_agents)
        result = depth, frozenset(agents)
        validated[version.id] = result
        return result

    issues = []
    for index, node in enumerate(Flow.model_validate(config["flow"]).nodes):
        if node.type != "handoff":
            continue
        try:
            call = HandoffConfig.model_validate(node.config)
            await check(call, definitions(config), (agent_id,))
        except ValidationError, KeyError:
            message = "Проверьте настройки блока передачи и целевого сценария"
        except ValueError as exc:
            message = str(exc)
        else:
            continue
        issues.append(
            Issue(("flow", "nodes", index, "config"), "invalid_handoff_dependency", message)
        )
    return issues, [version for version in versions.values() if version is not None]
