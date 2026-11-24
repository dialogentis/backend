"""Create a snapshot and its training job within the caller's transaction."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.agents import agent_and_draft
from dialogentis.api_errors import DomainValidationError
from dialogentis.datasets.routes import issue_view, validate_configuration
from dialogentis.drafts import DraftConfig
from dialogentis.flow_validation import validate_flow
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job
from dialogentis.models import User
from dialogentis.organizations import audit, organization_context
from dialogentis.versions.handoff_dependencies import validate_handoff_dependencies
from dialogentis.versions.models import AgentVersion


class TrainingInProgress(Exception):
    def __init__(self, job_id: UUID):
        self.job_id = job_id


@dataclass(frozen=True)
class PreparedVersion:
    version: AgentVersion
    replayed: bool


def canonical_hash(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def nlu_hash(snapshot: dict) -> str:
    config = snapshot["config"]
    return canonical_hash(
        {
            "model": snapshot["llm_model"],
            "intents": config["intents"],
            "entities": config["entities"],
            "examples": [item for item in config["examples"] if item["split"] == "train"],
        }
    )


type VersionStatus = Literal["building", "ready", "failed", "cancelled", "obsolete"]


def version_status(version: AgentVersion, job: Job) -> VersionStatus:
    if (
        job.id != version.job_id
        or job.organization_id != version.organization_id
        or job.kind != "agent.training"
    ):
        raise ValueError("Job does not belong to this version")
    if job.status in {"queued", "running"}:
        return "building"
    if job.status == "cancelled":
        return "cancelled"
    if (
        job.status == "succeeded"
        and version.model_artifact_id is not None
        and version.validation_report is not None
    ):
        return "ready" if version.snapshot.get("llm_model") else "obsolete"
    return "failed"


async def prepare_version(
    db: AsyncSession,
    user: User,
    org_id: UUID,
    agent_id: UUID,
    source_revision: int,
    idempotency_key: str,
    llm_model: str,
) -> PreparedVersion:
    if type(source_revision) is not int or source_revision < 1:
        raise ValueError("Invalid source revision")
    if not 1 <= len(idempotency_key) <= 128 or any(
        not 33 <= ord(char) <= 126 for char in idempotency_key
    ):
        raise ValueError("Invalid idempotency key")
    if not user.email_verified:
        raise HTTPException(403, "Подтвердите адрес почты")
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    previous = await db.scalar(
        select(AgentVersion).where(
            AgentVersion.organization_id == org_id,
            AgentVersion.agent_id == agent_id,
            AgentVersion.created_by == user.id,
            AgentVersion.idempotency_key_hash == key_hash,
        )
    )
    if previous is not None:
        if previous.source_revision != source_revision:
            raise HTTPException(409, "Ключ уже использован для другой ревизии")
        return PreparedVersion(previous, True)
    agent, draft = await agent_and_draft(db, org_id, agent_id)
    if agent.status != "active":
        raise HTTPException(409, "Для архивного агента нельзя подготовить версию")
    if draft.revision != source_revision:
        raise HTTPException(409, "Черновик изменился. Обновите страницу")
    running = await db.scalar(
        select(Job.id)
        .join(AgentVersion, AgentVersion.job_id == Job.id)
        .where(
            AgentVersion.organization_id == org_id,
            AgentVersion.agent_id == agent_id,
            Job.organization_id == org_id,
            Job.status.in_(("queued", "running")),
        )
    )
    if running is not None:
        raise TrainingInProgress(running)
    try:
        config = DraftConfig.model_validate(draft.configuration).model_dump(mode="json")
        encoded = json.dumps(config, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValidationError, ValueError, UnicodeError) as exc:
        raise HTTPException(422, "Проверьте данные черновика") from exc
    if len(encoded) > 10 * 1024 * 1024:
        raise HTTPException(413, "Черновик превышает 10 МиБ")
    errors, warnings = await asyncio.to_thread(validate_configuration, config)
    flow_errors, flow_warnings = await asyncio.to_thread(validate_flow, config)
    errors.extend(flow_errors)
    warnings.extend(flow_warnings)
    if not errors and any(node["type"] == "handoff" for node in config["flow"]["nodes"]):
        errors.extend(await validate_handoff_dependencies(db, org_id, agent_id, config))
    if errors:
        raise DomainValidationError([issue_view(item).model_dump() for item in errors[:1000]])
    if not llm_model:
        raise HTTPException(409, "Администратор сервера должен настроить подключение к LLM")
    snapshot = {"schema_version": draft.schema_version, "config": config, "llm_model": llm_model}
    number = await db.scalar(
        select(func.coalesce(func.max(AgentVersion.number), 0)).where(
            AgentVersion.organization_id == org_id, AgentVersion.agent_id == agent_id
        )
    )
    job = enqueue(db, org_id, "agent.training")
    await db.flush()
    version = AgentVersion(
        organization_id=org_id,
        agent_id=agent_id,
        number=int(number or 0) + 1,
        source_revision=source_revision,
        snapshot=snapshot,
        config_hash=canonical_hash(snapshot),
        nlu_hash=nlu_hash(snapshot),
        dataset_warnings=[issue_view(item).model_dump() for item in warnings[:1000]],
        warnings_truncated=len(warnings) > 1000,
        job_id=job.id,
        idempotency_key_hash=key_hash,
        created_by=user.id,
    )
    db.add(version)
    await db.flush()
    audit(db, org_id, user, "agent.training_requested", version.id)
    return PreparedVersion(version, False)
