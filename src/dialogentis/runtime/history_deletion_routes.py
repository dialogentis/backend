from datetime import datetime
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, insert, literal, select

from dialogentis.auth.security import VerifiedUser, digest
from dialogentis.contracts import InputModel
from dialogentis.datasets.export_routes import JobStatus
from dialogentis.dependencies import Database
from dialogentis.jobs.enqueue import enqueue
from dialogentis.jobs.models import Job
from dialogentis.models import Organization
from dialogentis.organizations import audit, organization_context
from dialogentis.runtime.history_deletion import count_changes
from dialogentis.runtime.history_deletion_models import HistoryDeletionTarget, HistoryDeletionTask
from dialogentis.runtime.history_filters import HistoryFilters, filtered_history
from dialogentis.runtime.history_models import ConversationHistory

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/history/deletions", tags=["history-deletions"]
)


class HistoryDeletionInput(InputModel):
    confirmation: Literal["delete"]
    filters: HistoryFilters = Field(default_factory=HistoryFilters)


class HistoryDeletionView(BaseModel):
    job_id: UUID
    status: JobStatus
    error_code: str | None
    requested_at: datetime
    selected_dialogues: int
    deleted_dialogues: int | None
    deleted_turns: int | None
    cleared_receipts: int | None


def view(job: Job, task: HistoryDeletionTask) -> HistoryDeletionView:
    return HistoryDeletionView(
        job_id=job.id,
        status=cast(JobStatus, job.status),
        error_code=job.error_code,
        requested_at=task.requested_at,
        selected_dialogues=task.selected_dialogues,
        deleted_dialogues=task.deleted_dialogues,
        deleted_turns=task.deleted_turns,
        cleared_receipts=task.cleared_receipts,
    )


@router.post("", status_code=202, response_model=HistoryDeletionView)
async def create_history_deletion(
    org_id: UUID,
    body: HistoryDeletionInput,
    user: VerifiedUser,
    db: Database,
    request_key: Annotated[UUID | None, Header(alias="Idempotency-Key")] = None,
):
    await organization_context(db, user, org_id, "manage_history", lock=True)
    start, end = body.filters.start_at, body.filters.end_at
    if start is not None and end is not None and start >= end:
        raise HTTPException(422, "Дата начала должна предшествовать окончанию периода")
    org = await db.get(Organization, org_id)
    assert org is not None
    filter_hash = digest(body.filters.model_dump_json(by_alias=True, exclude_none=True))
    if request_key is not None:
        existing = await db.scalar(
            select(HistoryDeletionTask).where(
                HistoryDeletionTask.organization_id == org_id,
                HistoryDeletionTask.requested_by == user.id,
                HistoryDeletionTask.request_key == request_key,
            )
        )
        if existing is not None:
            if existing.filter_hash != filter_hash:
                raise HTTPException(409, "Ключ операции уже использован с другими фильтрами")
            job = await db.get(Job, existing.job_id)
            assert job is not None
            return view(job, existing)
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    job = enqueue(db, org_id, "history.deletion")
    await db.flush()
    task = HistoryDeletionTask(
        job_id=job.id,
        organization_id=org_id,
        requested_by=user.id,
        requested_at=now,
        selected_dialogues=0,
        request_key=request_key,
        filter_hash=filter_hash,
    )
    db.add(task)
    await db.flush()
    targets = filtered_history(org, body.filters).with_only_columns(
        ConversationHistory.organization_id,
        literal(job.id),
        ConversationHistory.id,
    )
    task.selected_dialogues = await count_changes(
        db,
        insert(HistoryDeletionTarget)
        .from_select(
            ("organization_id", "job_id", "conversation_id"),
            targets,
        )
        .returning(HistoryDeletionTarget.conversation_id),
    )
    audit(
        db,
        org_id,
        user,
        "history.deletion_requested",
        job.id,
        details={"selected_dialogues": task.selected_dialogues},
    )
    return view(job, task)


@router.get("/{deletion_id}", response_model=HistoryDeletionView)
async def history_deletion_status(
    org_id: UUID, deletion_id: UUID, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_history")
    job = await db.scalar(
        select(Job).where(
            Job.organization_id == org_id,
            Job.id == deletion_id,
            Job.kind == "history.deletion",
        )
    )
    task = await db.get(HistoryDeletionTask, deletion_id) if job is not None else None
    if task is None or job is None:
        raise HTTPException(404, "Задание не найдено")
    return view(job, task)
