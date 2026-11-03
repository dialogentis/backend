from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from cryptography.fernet import InvalidToken
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from dialogentis.api_errors import error_response
from dialogentis.auth.models import RateEvent
from dialogentis.auth.security import VerifiedUser, check_password, rate_limit
from dialogentis.contracts import InputModel
from dialogentis.dependencies import Database
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.legacy_mail import has_untagged_invitation_mail
from dialogentis.models import Organization
from dialogentis.organizations import OrganizationDeletion, audit, organization_context

router = APIRouter(prefix="/api/v1", tags=["organizations"])


class DeletionInput(InputModel):
    password: str = Field(min_length=12, max_length=128)
    confirmation_name: str = Field(min_length=1, max_length=120)


class DeletionView(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    error_code: str | None


class DeletionRetryInput(InputModel):
    password: str = Field(min_length=12, max_length=128)


async def ensure_legacy_invitation_mail_is_drained(db: Database, request: Request) -> None:
    key = request.app.state.settings.encryption_key
    if key is None:
        raise HTTPException(503, "Удаление временно недоступно")
    try:
        if await has_untagged_invitation_mail(db, key.get_secret_value()):
            raise HTTPException(503, "Дождитесь отправки ранее поставленных в очередь писем")
    except (InvalidToken, ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(503, "Удаление временно недоступно") from exc


@router.post("/organizations/{org_id}/deletion", status_code=202, response_model=DeletionView)
async def request_deletion(
    org_id: UUID, body: DeletionInput, request: Request, user: VerifiedUser, db: Database
):
    await organization_context(
        db, user, org_id, "delete_organization", lock=True, allow_deleting=True
    )
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Пространство не найдено")
    key = await rate_limit(db, request, "delete-password", str(user.id), 5, 15, record=False)
    if not await check_password(user.password_hash, body.password):
        db.add(RateEvent(subject_hash=key))
        return error_response(request, 403, "invalid_password", "Неверный пароль")
    if body.confirmation_name != org.name:
        raise HTTPException(422, "Название пространства не совпадает")
    if org.deletion_requested_at is not None:
        existing = await db.scalar(
            select(OrganizationDeletion).where(OrganizationDeletion.organization_id == org_id)
        )
        if existing is None or existing.requested_by != user.id:
            raise HTTPException(404, "Пространство не найдено")
        job = await db.get(Job, existing.job_id)
        if job is None:
            raise HTTPException(503, "Состояние удаления недоступно")
        return {"job_id": job.id, "status": job.status, "error_code": job.error_code}
    await ensure_legacy_invitation_mail_is_drained(db, request)
    now = datetime.now(UTC)
    org.deletion_requested_at = now
    job = enqueue(db, org_id, "organization.deletion")
    db.add(
        OrganizationDeletion(
            job_id=job.id, organization_id=org_id, requested_by=user.id, requested_at=now
        )
    )
    audit(db, org_id, user, "organization.deletion_requested", job.id)
    return {"job_id": job.id, "status": "queued", "error_code": None}


@router.get("/deletions/{job_id}", response_model=DeletionView)
async def deletion_status(job_id: UUID, user: VerifiedUser, db: Database):
    record = await db.get(OrganizationDeletion, job_id)
    if record is None or record.requested_by != user.id:
        raise HTTPException(404, "Задание не найдено")
    await db.execute(
        text("SELECT set_config('app.organization_id', :id, true)"),
        {"id": str(record.organization_id)},
    )
    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Задание не найдено")
    return {"job_id": job.id, "status": job.status, "error_code": job.error_code}


@router.post("/deletions/{job_id}/retry", status_code=202, response_model=DeletionView)
async def retry_deletion(
    job_id: UUID, body: DeletionRetryInput, request: Request, user: VerifiedUser, db: Database
):
    record = await db.get(OrganizationDeletion, job_id)
    if record is None or record.requested_by != user.id:
        raise HTTPException(404, "Задание не найдено")
    await db.execute(
        text("SELECT set_config('app.organization_id', :id, true)"),
        {"id": str(record.organization_id)},
    )
    job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
    if job is None:
        raise HTTPException(404, "Задание не найдено")
    await organization_context(
        db, user, record.organization_id, "delete_organization", lock=True, allow_deleting=True
    )
    key = await rate_limit(db, request, "delete-password", str(user.id), 5, 15, record=False)
    if not await check_password(user.password_hash, body.password):
        db.add(RateEvent(subject_hash=key))
        return error_response(request, 403, "invalid_password", "Неверный пароль")
    if job.status in {"queued", "running"}:
        return {"job_id": job.id, "status": job.status, "error_code": job.error_code}
    if job.status != "failed":
        raise HTTPException(409, "Повтор удаления недоступен")
    job.status = "queued"
    job.attempts = job.delivery_recoveries = 0
    job.cancel_requested = False
    job.error_code = None
    job.finished_at = job.next_retry_at = None
    job.execution_token = None
    db.add(OutboxEvent(organization_id=record.organization_id, job_id=job.id))
    audit(db, record.organization_id, user, "organization.deletion_retried", job.id)
    return {"job_id": job.id, "status": "queued", "error_code": None}
