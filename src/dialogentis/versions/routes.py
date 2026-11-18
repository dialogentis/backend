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
from dialogentis.nlu.entity_evaluation import EntityMetrics, EntitySpan
from dialogentis.nlu.intent import FallbackReason
from dialogentis.nlu.intent_evaluation import ClassMetrics
from dialogentis.nlu.settings import NLUSettings, QualitySettings
from dialogentis.organizations import organization_context
from dialogentis.slots import validate_slots
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import TrainingInProgress, prepare_version, version_status

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/agents/{agent_id}/versions", tags=["versions"]
)
Stage = Literal["validation", "intent_training", "entity_training", "evaluation", "packaging"]
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
    graph_validation: Literal["not_checked"] | None


class HandoffSlotView(BaseModel):
    id: UUID
    code: str
    name: str
    value_type: Literal["string", "integer", "number", "boolean", "date"]
    sensitive: bool


class IntentSummary(BaseModel):
    labels: list[UUID | None]
    confusion_matrix: list[list[int]]
    test_count: int
    known_count: int
    unknown_count: int
    excluded_disabled_count: int
    accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    unknown_rejection: float | None
    known_fallback_rate: float | None
    per_intent: list[ClassMetrics]
    error_count: int


class EntitySummary(BaseModel):
    test_count: int
    excluded_disabled_count: int
    expected: int
    predicted: int
    true_positive: int
    precision: float | None
    recall: float | None
    f1: float | None
    invalid_values: int
    per_entity: list[EntityMetrics]
    error_count: int


class CatalogEntry(BaseModel):
    id: UUID
    name: str
    code: str


class ReportView(BaseModel):
    version_id: UUID
    training_examples: int
    nlu_settings: NLUSettings
    quality_settings: QualitySettings
    intents: IntentSummary
    entities: EntitySummary
    intent_catalog: list[CatalogEntry]
    entity_catalog: list[CatalogEntry]
    graph_validation: Literal["not_checked"]
    warnings: list[str]


class IntentErrorView(BaseModel):
    example_id: UUID
    text: str
    expected: UUID | None
    predicted: UUID | None
    fallback_reasons: list[FallbackReason]


class EntityErrorView(BaseModel):
    example_id: UUID
    text: str
    missing: list[EntitySpan]
    unexpected: list[EntitySpan]
    invalid_values: list[EntitySpan]


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
        graph_validation=(
            "not_checked"
            if version.validation_report
            and version.validation_report.get("graph_validation") == "not_checked"
            else None
        ),
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


def _report(version: AgentVersion, job: Job) -> dict:
    if version_status(version, job) != "ready" or version.validation_report is None:
        raise HTTPException(409, "Отчёт ещё не готов")
    return version.validation_report


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
        result = await prepare_version(db, user, org_id, agent_id, body.source_revision, key)
    except TrainingInProgress as exc:
        return error_response(
            request,
            409,
            "training_in_progress",
            "Агент уже обучается",
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


@router.get("/{version_id}/report", response_model=ReportView)
async def get_report(
    org_id: UUID, agent_id: UUID, version_id: UUID, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id)
    version, job = await _version(db, org_id, agent_id, version_id)
    report = _report(version, job)
    intent = {**report["intents"], "error_count": len(report["intents"]["errors"])}
    entity = {**report["entities"], "error_count": len(report["entities"]["errors"])}
    config = version.snapshot["config"]
    return ReportView(
        version_id=version.id,
        training_examples=report["training_examples"],
        nlu_settings=NLUSettings.model_validate(version.snapshot["nlu_settings"]),
        quality_settings=QualitySettings.model_validate(config.get("quality_settings", {})),
        intents=IntentSummary.model_validate(intent),
        entities=EntitySummary.model_validate(entity),
        intent_catalog=[CatalogEntry.model_validate(item) for item in config["intents"]],
        entity_catalog=[CatalogEntry.model_validate(item) for item in config["entities"]],
        graph_validation=report["graph_validation"],
        warnings=report["warnings"],
    )


@router.get(
    "/{version_id}/report/errors",
    response_model=CursorPage[IntentErrorView | EntityErrorView],
)
async def report_errors(
    org_id: UUID,
    agent_id: UUID,
    version_id: UUID,
    user: VerifiedUser,
    db: Database,
    kind: Literal["intent", "entity"],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[int, Query(ge=0, le=3000)] = 0,
):
    await organization_context(db, user, org_id)
    version, job = await _version(db, org_id, agent_id, version_id)
    errors = _report(version, job)["intents" if kind == "intent" else "entities"]["errors"]
    texts = {
        item["id"]: item["text"]
        for item in version.snapshot["config"]["examples"]
        if item["split"] == "test"
    }
    page = errors[cursor : cursor + limit]
    if any(item["example_id"] not in texts for item in page):
        raise HTTPException(503, "Данные отчёта временно недоступны")
    return {
        "items": [{**item, "text": texts[item["example_id"]]} for item in page],
        "next_cursor": str(cursor + limit) if cursor + limit < len(errors) else None,
    }
