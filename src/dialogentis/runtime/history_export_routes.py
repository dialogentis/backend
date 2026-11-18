from datetime import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from dialogentis.artifact_service import ArtifactUnavailable, read_artifact
from dialogentis.artifacts import ArtifactIntegrityError, ArtifactStorage
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.datasets.export_routes import JobStatus
from dialogentis.dependencies import Database
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job
from dialogentis.models import Organization
from dialogentis.organizations import audit, organization_context
from dialogentis.runtime.history_export import ENVELOPE_BYTES, MAX_EXPORT_BYTES, open_history_export
from dialogentis.runtime.history_export_models import HistoryExportTask, HistoryFormat
from dialogentis.runtime.history_filters import HistoryFilters
from dialogentis.runtime.receipts import ReceiptError

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/history/exports", tags=["history-exports"]
)


class HistoryExportInput(InputModel):
    format: HistoryFormat
    filters: HistoryFilters = Field(default_factory=HistoryFilters)


class HistoryExportView(BaseModel):
    job_id: UUID
    format: HistoryFormat
    status: JobStatus
    error_code: str | None = None
    download_until: datetime | None = None


def status_view(job: Job, task: HistoryExportTask, org: Organization) -> HistoryExportView:
    available = (
        job.status == "succeeded"
        and task.artifact_id is not None
        and task.settings_revision == org.settings_revision
    )
    return HistoryExportView(
        job_id=job.id,
        format=task.format,
        status=cast(JobStatus, job.status),
        error_code=job.error_code,
        download_until=task.download_until if available else None,
    )


@router.post("", status_code=202, response_model=HistoryExportView)
async def create_history_export(
    org_id: UUID, body: HistoryExportInput, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_history", lock=True)
    org = await db.get(Organization, org_id)
    assert org is not None
    start, end = body.filters.start_at, body.filters.end_at
    if start is not None and end is not None and start >= end:
        raise HTTPException(422, "Дата начала должна предшествовать окончанию периода")
    job = enqueue(db, org_id, "history.export")
    await db.flush()
    task = HistoryExportTask(
        job_id=job.id,
        organization_id=org_id,
        requested_by=user.id,
        format=body.format,
        filters=body.filters.model_dump(mode="json", by_alias=True, exclude_none=True),
        settings_revision=org.settings_revision,
    )
    db.add(task)
    audit(db, org_id, user, "history.export_requested", job.id, details={"format": body.format})
    return status_view(job, task, org)


async def find_export(db: Database, org_id: UUID, export_id: UUID):
    job = await db.scalar(
        select(Job).where(
            Job.organization_id == org_id,
            Job.id == export_id,
            Job.kind == "history.export",
        )
    )
    if job is None:
        raise HTTPException(404, "Задание не найдено")
    task = await db.get(HistoryExportTask, job.id)
    if task is None:
        raise HTTPException(404, "Задание не найдено")
    return job, task


@router.get("/{export_id}", response_model=HistoryExportView)
async def history_export_status(org_id: UUID, export_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_history")
    org = await db.get(Organization, org_id)
    assert org is not None
    job, task = await find_export(db, org_id, export_id)
    return status_view(job, task, org)


@router.get("/{export_id}/download")
async def download_history_export(
    org_id: UUID, export_id: UUID, request: Request, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_history", lock=True)
    org = await db.get(Organization, org_id)
    assert org is not None
    job, task = await find_export(db, org_id, export_id)
    now = await db.scalar(select(func.clock_timestamp()))
    if (
        job.status != "succeeded"
        or task.settings_revision != org.settings_revision
        or task.artifact_id is None
        or task.download_until is None
        or now is None
        or task.download_until <= now
    ):
        raise HTTPException(404, "Файл недоступен или срок скачивания истёк")
    settings = request.app.state.settings
    try:
        encrypted = await read_artifact(
            db,
            ArtifactStorage(settings.artifact_root),
            org_id,
            task.artifact_id,
            max_bytes=MAX_EXPORT_BYTES + ENVELOPE_BYTES,
            expected_kind="temporary",
        )
        content = open_history_export(encrypted, org_id, export_id, settings)
    except ArtifactUnavailable:
        raise HTTPException(404, "Файл недоступен или срок скачивания истёк") from None
    except OSError, ArtifactIntegrityError, ReceiptError:
        raise HTTPException(503, "Файл временно недоступен") from None
    media = "application/json" if task.format == "json" else "text/csv; charset=utf-8"
    audit(db, org_id, user, "history.export_downloaded", job.id, details={"format": task.format})
    return Response(
        content,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="dialogentis-history.{task.format}"'
        },
    )
