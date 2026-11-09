"""Create a snapshot and its training job within the caller's transaction."""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.agents import agent_and_draft
from dialogentis.api_errors import DomainValidationError
from dialogentis.datasets.routes import issue_view, validate_configuration
from dialogentis.datasets.schema import Dataset
from dialogentis.drafts import DraftConfig
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job
from dialogentis.models import User
from dialogentis.nlu.intent import Thresholds
from dialogentis.organizations import audit, organization_context
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


def nlu_hash(config: dict, settings: dict) -> str:
    data = Dataset.model_validate({key: config[key] for key in ("intents", "entities", "examples")})
    payload = data.model_dump(mode="json")
    payload["examples"] = [item for item in payload["examples"] if item["split"] == "train"]
    for key in ("intents", "entities", "examples"):
        payload[key].sort(key=lambda item: item["id"])
    return canonical_hash({"schema_version": 1, "dataset": payload, "settings": settings})


def version_status(
    version: AgentVersion, job: Job
) -> Literal["building", "ready", "failed", "cancelled"]:
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
        return "ready"
    return "failed"


async def prepare_version(
    db: AsyncSession,
    user: User,
    org_id: UUID,
    agent_id: UUID,
    source_revision: int,
    idempotency_key: str,
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
        raise HTTPException(409, "Архивного агента нельзя обучить")
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
    if errors:
        raise DomainValidationError([issue_view(item).model_dump() for item in errors[:1000]])
    settings = asdict(Thresholds())
    snapshot = {"schema_version": draft.schema_version, "config": config, "nlu_settings": settings}
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
        nlu_hash=nlu_hash(config, settings),
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
