from datetime import datetime
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, StrictInt
from sqlalchemy import select

from dialogentis.api_errors import error_response
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage, InputModel
from dialogentis.datasets.routes import ValidationIssue
from dialogentis.dependencies import Database
from dialogentis.jobs.models import Job
from dialogentis.models import Agent
from dialogentis.organizations import organization_context
from dialogentis.slots import validate_slots
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import TrainingInProgress, prepare_version, version_status

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/agents/{agent_id}/versions", tags=["versions"]
)
Stage = Literal["validation", "packaging"]
VersionStatus = Literal["building", "ready", "failed", "cancelled"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
IdempotencyKey = Annotated[
    str, Header(alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[!-~]+$")
]


class VersionInput(InputModel):
    source_revision: StrictInt = Field(ge=1)


class VersionCreated(BaseModel):
    version_id: UUID
    job_id: UUID
    replayed: bool


class VersionView(BaseModel):
    id: UUID
    job_id: UUID
    number: int
    source_revision: int
    status: VersionStatus
    job_status: JobStatus
    cancel_requested: bool
    error_code: str | None
    stage: Stage | None
    created_by: UUID
    created_at: datetime
    config_hash: str
    nlu_hash: str
    dataset_warnings: list[ValidationIssue]
    warnings_truncated: bool
    llm_model: str | None


class HandoffSlotView(BaseModel):
    id: UUID
    code: str
    name: str
    value_type: Literal["string", "integer", "number", "boolean", "date"]
    sensitive: bool


def view(version: AgentVersion, job: Job) -> VersionView:
    return VersionView(
        id=version.id,
        job_id=job.id,
        number=version.number,
        source_revision=version.source_revision,
        status=version_status(version, job),
        job_status=cast(JobStatus, job.status),
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        stage=cast(Stage | None, job.stage),
        created_by=version.created_by,
        created_at=version.created_at,
        config_hash=version.config_hash,
        nlu_hash=version.nlu_hash,
        dataset_warnings=version.dataset_warnings,
        warnings_truncated=version.warnings_truncated,
        llm_model=version.snapshot.get("llm_model"),
    )


async def _version(db: Database, org_id: UUID, agent_id: UUID, version_id: UUID):
    row = (
        await db.execute(
            select(AgentVersion, Job)
            .join(Job, Job.id == AgentVersion.job_id)
            .where(
                AgentVersion.id == version_id,
                AgentVersion.organization_id == org_id,
                AgentVersion.agent_id == agent_id,
                Job.organization_id == org_id,
            )
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(404, "Версия не найдена")
    return row


@router.post("", status_code=202, response_model=VersionCreated)
async def create_version(
    org_id: UUID,
    agent_id: UUID,
    body: VersionInput,
    request: Request,
    user: VerifiedUser,
    db: Database,
    key: IdempotencyKey,
):
    try:
        result = await prepare_version(
            db,
            user,
            org_id,
            agent_id,
            body.source_revision,
            key,
            request.app.state.settings.llm_model,
        )
    except TrainingInProgress as exc:
        return error_response(
            request,
            409,
            "training_in_progress",
            "Версия агента уже готовится",
            [{"job_id": str(exc.job_id)}],
        )
    return VersionCreated(
        version_id=result.version.id, job_id=result.version.job_id, replayed=result.replayed
    )


@router.get("", response_model=CursorPage[VersionView])
async def list_versions(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[int | None, Query(ge=1)] = None,
):
    await organization_context(db, user, org_id)
    agent = await db.scalar(
        select(Agent.id).where(Agent.id == agent_id, Agent.organization_id == org_id)
    )
    if agent is None:
        raise HTTPException(404, "Агент не найден")
    query = (
        select(AgentVersion, Job)
        .join(Job, Job.id == AgentVersion.job_id)
        .where(
            AgentVersion.organization_id == org_id,
            AgentVersion.agent_id == agent_id,
            Job.organization_id == org_id,
        )
        .order_by(AgentVersion.number.desc())
        .limit(limit + 1)
    )
    if cursor is not None:
        query = query.where(AgentVersion.number < cursor)
    rows = (await db.execute(query)).all()
    return {
        "items": [view(version, job) for version, job in rows[:limit]],
        "next_cursor": str(rows[limit - 1][0].number) if len(rows) > limit else None,
    }


@router.get("/{version_id}", response_model=VersionView)
async def get_version(
    org_id: UUID, agent_id: UUID, version_id: UUID, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id)
    version, job = await _version(db, org_id, agent_id, version_id)
    return view(version, job)


@router.get("/{version_id}/handoff-slots", response_model=list[HandoffSlotView])
async def list_handoff_slots(
    org_id: UUID, agent_id: UUID, version_id: UUID, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id)
    version, job = await _version(db, org_id, agent_id, version_id)
    if version_status(version, job) != "ready":
        raise HTTPException(409, "Версия агента ещё не готова")
    slots, errors = validate_slots(version.snapshot["config"].get("slots", []))
    if errors:
        raise HTTPException(409, "Проверьте параметры версии")
    return [HandoffSlotView.model_validate(slot, from_attributes=True) for slot in slots]
