"""Readiness checks for a proposed immutable release."""

import asyncio
from collections import Counter
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.artifact_records import Artifact
from dialogentis.config import Settings
from dialogentis.datasets.routes import validate_configuration
from dialogentis.datasets.schema import Dataset
from dialogentis.flow_validation import validate_flow
from dialogentis.http_action import HttpActionConfig
from dialogentis.integrations.models import Integration
from dialogentis.integrations.schema import path_allowed
from dialogentis.integrations.secrets import open_secret
from dialogentis.jobs.models import Job
from dialogentis.models import Agent
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.versions.handoff_dependencies import resolve_handoff_dependencies
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import canonical_hash, version_status


class ReleasePin(BaseModel):
    agent_id: UUID
    version_id: UUID
    model_artifact_id: UUID
    config_hash: str


class ReleaseIssue(BaseModel):
    code: str
    message: str
    version_id: UUID | None = None


class ReleaseCheck(BaseModel):
    errors: list[ReleaseIssue] = []
    warnings: list[ReleaseIssue] = []
    manifest: list[ReleasePin] = []


async def _has_smoke(db: AsyncSession, agent: Agent, version_id: UUID) -> bool:
    return (
        await db.scalar(
            select(RuntimeReceipt.id)
            .join(
                TestConversation,
                and_(
                    TestConversation.id == RuntimeReceipt.conversation_id,
                    TestConversation.organization_id == RuntimeReceipt.organization_id,
                ),
            )
            .where(
                TestConversation.organization_id == agent.organization_id,
                TestConversation.agent_id == agent.id,
                TestConversation.version_id == version_id,
                TestConversation.http_mode == "mock",
                RuntimeReceipt.status == "completed",
                or_(
                    RuntimeReceipt.runtime_status == "completed",
                    and_(
                        RuntimeReceipt.runtime_status == "waiting_input",
                        RuntimeReceipt.revision > 1,
                    ),
                ),
            )
            .limit(1)
        )
    ) is not None


def _configuration(version: AgentVersion, check: ReleaseCheck) -> list[HttpActionConfig]:
    config = version.snapshot["config"]
    errors, _ = validate_configuration(config)
    graph_errors, _ = validate_flow(config)
    for issue in [*errors, *graph_errors][:100]:
        check.errors.append(
            ReleaseIssue(code=issue.code, message=issue.message, version_id=version.id)
        )
    if errors or graph_errors:
        return []

    dataset = Dataset.model_validate(
        {key: config[key] for key in ("intents", "entities", "examples")}
    )
    test = [item for item in dataset.examples if item.split == "test"]
    counts = Counter(item.intent_id for item in test)
    if any(intent.enabled and counts[intent.id] < 3 for intent in dataset.intents):
        check.errors.append(
            ReleaseIssue(
                code="release_test_minimum",
                message="Нужны хотя бы три тестовые фразы на каждое включённое намерение",
                version_id=version.id,
            )
        )
    report = version.validation_report or {}
    for group in ("intents", "entities"):
        metrics = report.get(group, {})
        evaluated, excluded = metrics.get("test_count"), metrics.get("excluded_disabled_count")
        if (
            type(evaluated) is not int
            or type(excluded) is not int
            or evaluated + excluded != len(test)
        ):
            check.errors.append(
                ReleaseIssue(
                    code="release_incomplete_evaluation",
                    message="Отчёт не подтверждает оценку всех тестовых примеров",
                    version_id=version.id,
                )
            )
            break
    for kind, name in (("intent", "намерений"), ("entity", "сущностей")):
        if f"low_{kind}_quality" in report.get("warnings", []):
            check.warnings.append(
                ReleaseIssue(
                    code=f"low_{kind}_quality",
                    message=f"F1 {name} ниже порога, указанного в версии",
                    version_id=version.id,
                )
            )
    return [
        HttpActionConfig.model_validate(node["config"])
        for node in config["flow"]["nodes"]
        if node["type"] == "http_action"
    ]


async def check_release(
    db: AsyncSession, agent: Agent, version_id: UUID, settings: Settings
) -> ReleaseCheck:
    org_id = agent.organization_id
    row = (
        await db.execute(
            select(AgentVersion, Job)
            .join(Job, and_(Job.id == AgentVersion.job_id, Job.organization_id == org_id))
            .where(
                AgentVersion.organization_id == org_id,
                AgentVersion.agent_id == agent.id,
                AgentVersion.id == version_id,
            )
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(404, "Версия не найдена")
    root, job = row
    check = ReleaseCheck()
    if agent.status != "active" or version_status(root, job) != "ready":
        check.errors.append(
            ReleaseIssue(
                code="release_not_ready",
                message="Нужны активный агент и готовая версия",
                version_id=version_id,
            )
        )
        return check

    actions = await asyncio.to_thread(_configuration, root, check)
    if check.errors:
        return check
    issues, dependencies = await resolve_handoff_dependencies(
        db, org_id, agent.id, root.snapshot["config"]
    )
    if issues:
        check.errors.extend(
            ReleaseIssue(code=issue.code, message=issue.message, version_id=root.id)
            for issue in issues
        )
        return check
    for dependency in dependencies:
        actions.extend(await asyncio.to_thread(_configuration, dependency, check))
    versions = [root, *dependencies]
    active_ids = set(
        (
            await db.scalars(
                select(Agent.id).where(
                    Agent.organization_id == org_id,
                    Agent.id.in_({v.agent_id for v in versions}),
                    Agent.status == "active",
                )
            )
        ).all()
    )
    artifact_ids = set(
        (
            await db.scalars(
                select(Artifact.id).where(
                    Artifact.organization_id == org_id,
                    Artifact.id.in_({v.model_artifact_id for v in versions}),
                    Artifact.kind == "model",
                    Artifact.status == "ready",
                )
            )
        ).all()
    )
    for version in versions:
        artifact_id = version.model_artifact_id
        if (
            version.agent_id not in active_ids
            or artifact_id not in artifact_ids
            or artifact_id is None
        ):
            check.errors.append(
                ReleaseIssue(
                    code="release_unavailable_dependency",
                    message="Версия или модель недоступна",
                    version_id=version.id,
                )
            )
            continue
        if await asyncio.to_thread(canonical_hash, version.snapshot) != version.config_hash:
            check.errors.append(
                ReleaseIssue(
                    code="release_invalid_snapshot",
                    message="Снимок версии повреждён",
                    version_id=version.id,
                )
            )
            continue
        check.manifest.append(
            ReleasePin(
                agent_id=version.agent_id,
                version_id=version.id,
                model_artifact_id=artifact_id,
                config_hash=version.config_hash,
            )
        )
    policies = {
        item.id: item
        for item in (
            await db.scalars(
                select(Integration).where(
                    Integration.organization_id == org_id,
                    Integration.id.in_({action.integration_id for action in actions}),
                )
            )
        ).all()
    }
    invalid: set[UUID] = set()
    for action in actions:
        policy = policies.get(action.integration_id)
        try:
            if (
                policy is None
                or not policy.enabled
                or action.method not in policy.methods
                or not path_allowed(action.path, policy.paths)
                or set(action.response_mapping) - set(policy.response_fields)
            ):
                raise ValueError("Unavailable integration")
            if policy.encrypted_secret is not None:
                open_secret(policy.encrypted_secret, org_id, policy.id, settings)
        except ValueError, ValidationError:
            invalid.add(action.integration_id)
    for _ in invalid:
        check.errors.append(
            ReleaseIssue(
                code="release_invalid_integration",
                message="Проверьте настройки интеграции HTTP-блока",
            )
        )
    if not await _has_smoke(db, agent, version_id):
        check.errors.append(
            ReleaseIssue(
                code="release_smoke_required",
                message="Проведите успешный диалог с тестовыми HTTP-ответами для этой версии",
                version_id=version_id,
            )
        )
    return check
