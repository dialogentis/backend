import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select

from dialogentis.agents import agent_and_draft
from dialogentis.artifact_service import ArtifactQuotaExceeded, reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import revision_etag
from dialogentis.datasets.csv_import import MAX_FILE_BYTES
from dialogentis.datasets.import_models import ImportTask
from dialogentis.datasets.routes import validate_configuration
from dialogentis.dependencies import Database
from dialogentis.drafts import (
    DraftConfig,
    DraftView,
    IfMatch,
    LeaseFence,
    LeaseToken,
    check_lease,
    draft_view,
)
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job
from dialogentis.organizations import audit, organization_context
from dialogentis.slots import removed_slot_references

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["imports"])
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class ImportCreated(BaseModel):
    job_id: UUID
    preview_id: UUID
    base_revision: int


class ImportIssue(BaseModel):
    row: int | None
    field: str
    path: list[str | int]
    code: str


class ImportReport(BaseModel):
    errors: list[ImportIssue]
    warnings: list[ImportIssue]
    counts: dict[str, int]
    new_codes: dict[str, list[str]]
    existing_codes: dict[str, list[str]]
    codes_truncated: bool = False


class ImportView(BaseModel):
    preview_id: UUID
    job_id: UUID
    status: JobStatus
    cancel_requested: bool
    error_code: str | None
    format: Literal["csv", "json"]
    mode: Literal["merge", "replace"]
    base_revision: int
    expires_at: datetime
    applied_revision: int | None
    can_apply: bool
    report: ImportReport | None


@router.post("/imports/preview", status_code=202, response_model=ImportCreated)
async def create_import(
    org_id: UUID,
    agent_id: UUID,
    request: Request,
    user: VerifiedUser,
    db: Database,
    file: Annotated[UploadFile, File()],
    mode: Annotated[Literal["merge", "replace"], Form()] = "merge",
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    form = await request.form()
    if (
        set(form.keys()) - {"file", "mode"}
        or len(form.getlist("file")) != 1
        or len(form.getlist("mode")) > 1
    ):
        raise HTTPException(422, "Некорректные поля загрузки")
    filename = (file.filename or "").lower()
    if filename.endswith(".csv"):
        format = "csv"
    elif filename.endswith(".json"):
        format = "json"
    else:
        raise HTTPException(422, "Выберите файл CSV или JSON")
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, "Файл превышает 10 МиБ")
    _, draft = await agent_and_draft(db, org_id, agent_id)
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    job = enqueue(db, org_id, "dataset.import")
    await db.flush()
    try:
        artifact = await reserve_artifact(
            db,
            org_id,
            kind="temporary",
            max_bytes=len(content),
            settings=request.app.state.settings,
            job_id=job.id,
        )
        await write_artifact(
            db,
            ArtifactStorage(request.app.state.settings.artifact_root),
            org_id,
            artifact.id,
            [content],
        )
    except ArtifactQuotaExceeded as exc:
        raise HTTPException(409, "Недостаточно места для импорта") from exc
    except OSError as exc:
        raise HTTPException(503, "Хранилище временно недоступно") from exc
    db.add(
        ImportTask(
            job_id=job.id,
            organization_id=org_id,
            agent_id=agent_id,
            requested_by=user.id,
            format=format,
            mode=mode,
            base_revision=draft.revision,
            input_artifact_id=artifact.id,
            expires_at=now + timedelta(hours=1),
        )
    )
    audit(db, org_id, user, "agent.import_requested", job.id)
    await db.flush()
    return ImportCreated(job_id=job.id, preview_id=job.id, base_revision=draft.revision)


async def import_task(
    db: Database, org_id: UUID, agent_id: UUID, preview_id: UUID, *, lock: bool = False
):
    query = (
        select(ImportTask, Job)
        .join(Job, Job.id == ImportTask.job_id)
        .where(
            ImportTask.job_id == preview_id,
            ImportTask.organization_id == org_id,
            ImportTask.agent_id == agent_id,
            Job.organization_id == org_id,
            Job.kind == "dataset.import",
        )
    )
    if lock:
        query = query.with_for_update(of=ImportTask)
    row = (await db.execute(query)).one_or_none()
    if row is None:
        raise HTTPException(404, "Проверка импорта не найдена")
    return row


@router.get("/imports/{preview_id}", response_model=ImportView)
async def get_import(
    org_id: UUID, agent_id: UUID, preview_id: UUID, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_agents")
    task, job = await import_task(db, org_id, agent_id, preview_id)
    now = await db.scalar(select(func.clock_timestamp()))
    if now is None or task.expires_at <= now:
        raise HTTPException(404, "Срок проверки импорта истёк")
    return ImportView(
        preview_id=task.job_id,
        job_id=job.id,
        status=cast(JobStatus, job.status),
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        format=task.format,
        mode=task.mode,
        base_revision=task.base_revision,
        expires_at=task.expires_at,
        applied_revision=task.applied_revision,
        can_apply=job.status == "succeeded"
        and task.candidate is not None
        and task.applied_revision is None,
        report=ImportReport.model_validate(task.report) if task.report is not None else None,
    )


@router.post("/imports/{preview_id}/apply", response_model=DraftView)
async def apply_import(
    org_id: UUID,
    agent_id: UUID,
    preview_id: UUID,
    user: VerifiedUser,
    db: Database,
    response: Response,
    token: LeaseToken,
    fence: LeaseFence,
    if_match: IfMatch,
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    task, job = await import_task(db, org_id, agent_id, preview_id, lock=True)
    now = await db.scalar(select(func.clock_timestamp()))
    if now is None or task.expires_at <= now:
        raise HTTPException(404, "Срок проверки импорта истёк")
    if job.status != "succeeded" or task.candidate is None or task.applied_revision is not None:
        raise HTTPException(409, "Проверенный импорт недоступен")
    _, draft = await agent_and_draft(db, org_id, agent_id)
    check_lease(draft, user, token, fence, datetime.now(UTC))
    if draft.revision != task.base_revision:
        raise HTTPException(409, "Черновик изменился. Проверьте файл заново")
    if if_match != revision_etag(draft.revision):
        raise HTTPException(412, "Черновик уже изменён. Обновите страницу")
    try:
        config = DraftConfig.model_validate(task.candidate).model_dump(mode="json")
        size = len(json.dumps(config, ensure_ascii=False, allow_nan=False).encode())
    except (ValidationError, ValueError, UnicodeError) as exc:
        raise HTTPException(409, "Проверенный импорт больше недействителен") from exc
    if size > MAX_FILE_BYTES:
        raise HTTPException(413, "Черновик превышает 10 МиБ")
    errors, _ = await asyncio.to_thread(validate_configuration, config)
    errors.extend(removed_slot_references(draft.configuration, config))
    readiness = {
        "insufficient_intents",
        "insufficient_intent_examples",
        "insufficient_entity_examples",
    }
    if any(issue.code not in readiness for issue in errors):
        raise HTTPException(409, "Проверенный импорт больше недействителен")
    draft.configuration = config
    draft.revision += 1
    task.applied_revision, task.candidate = draft.revision, None
    audit(db, org_id, user, "agent.import_applied", task.job_id)
    await db.flush()
    response.headers["ETag"] = revision_etag(draft.revision)
    return await draft_view(draft, db)
