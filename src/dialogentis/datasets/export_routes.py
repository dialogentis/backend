from datetime import timedelta
from typing import Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select

from dialogentis.agents import agent_and_draft
from dialogentis.artifact_service import ArtifactUnavailable, read_artifact
from dialogentis.artifacts import ArtifactIntegrityError, ArtifactStorage
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.datasets.export_models import ExportTask
from dialogentis.dependencies import Database
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.lifecycle import finish
from dialogentis.jobs.models import Job
from dialogentis.organizations import audit, organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}", tags=["exports"])
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class ExportInput(InputModel):
    format: Literal["json", "csv"]
    source: Literal["draft", "version"] = "draft"
    version_id: UUID | None = None


class ExportCreated(BaseModel):
    job_id: UUID
    export_id: UUID
    revision: int


class ExportJob(BaseModel):
    job_id: UUID
    status: JobStatus
    cancel_requested: bool
    error_code: str | None
    export_id: UUID
    revision: int
    format: Literal["json", "csv"]
    download_until: str | None


@router.post("/agents/{agent_id}/exports", status_code=202, response_model=ExportCreated)
async def create_export(
    org_id: UUID, agent_id: UUID, body: ExportInput, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    if body.source != "draft" or body.version_id is not None:
        raise HTTPException(422, "Экспорт версий пока недоступен")
    _, draft = await agent_and_draft(db, org_id, agent_id)
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    job = enqueue(db, org_id, "dataset.export")
    await db.flush()
    db.add(
        ExportTask(
            job_id=job.id,
            organization_id=org_id,
            agent_id=agent_id,
            requested_by=user.id,
            format=body.format,
            revision=draft.revision,
            snapshot=draft.configuration,
            snapshot_expires_at=now + timedelta(hours=24),
        )
    )
    audit(db, org_id, user, "agent.export_requested", job.id)
    await db.flush()
    return ExportCreated(job_id=job.id, export_id=job.id, revision=draft.revision)


async def export_and_job(db: Database, org_id: UUID, job_id: UUID, *, lock: bool = False):
    query = (
        select(Job, ExportTask)
        .join(ExportTask, ExportTask.job_id == Job.id)
        .where(
            Job.id == job_id,
            Job.organization_id == org_id,
            ExportTask.organization_id == org_id,
            Job.kind == "dataset.export",
        )
    )
    if lock:
        query = query.with_for_update(of=Job)
    row = (await db.execute(query)).one_or_none()
    if row is None:
        raise HTTPException(404, "Задание не найдено")
    return row


def job_view(job: Job, task: ExportTask) -> ExportJob:
    return ExportJob(
        job_id=job.id,
        status=cast(JobStatus, job.status),
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        export_id=job.id,
        revision=task.revision,
        format=task.format,
        download_until=task.download_until.isoformat() if task.download_until else None,
    )


@router.get("/jobs/{job_id}", response_model=ExportJob)
async def get_export_job(org_id: UUID, job_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_agents")
    job, task = await export_and_job(db, org_id, job_id)
    return job_view(job, task)


@router.post("/jobs/{job_id}/cancel", response_model=ExportJob)
async def cancel_export_job(org_id: UUID, job_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_agents")
    job, task = await export_and_job(db, org_id, job_id, lock=True)
    member = await organization_context(db, user, org_id, "manage_agents", lock=True)
    if task.requested_by != user.id and member.role not in {"owner", "admin"}:
        raise HTTPException(403, "Отмена доступна автору задания или администратору")
    if job.status not in {"queued", "running"}:
        raise HTTPException(409, "Задание уже завершено")
    job.cancel_requested = True
    if job.status == "queued":
        now = await db.scalar(select(func.clock_timestamp()))
        assert now is not None
        finish(job, "cancelled", now)
        task.snapshot = None
    return job_view(job, task)


@router.get("/exports/{export_id}/download")
async def download_export(
    org_id: UUID, export_id: UUID, request: Request, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    job, task = await export_and_job(db, org_id, export_id)
    now = await db.scalar(select(func.clock_timestamp()))
    if (
        job.status != "succeeded"
        or task.artifact_id is None
        or task.download_until is None
        or now is None
        or task.download_until <= now
    ):
        raise HTTPException(404, "Файл недоступен или срок скачивания истёк")
    try:
        content = await read_artifact(
            db,
            ArtifactStorage(request.app.state.settings.artifact_root),
            org_id,
            task.artifact_id,
            max_bytes=10 * 1024 * 1024,
        )
    except ArtifactUnavailable as exc:
        raise HTTPException(404, "Файл недоступен или срок скачивания истёк") from exc
    except (OSError, ArtifactIntegrityError) as exc:
        raise HTTPException(503, "Файл временно недоступен") from exc
    media = "application/json" if task.format == "json" else "text/csv; charset=utf-8"
    return Response(
        content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="dialogentis.{task.format}"'},
    )
