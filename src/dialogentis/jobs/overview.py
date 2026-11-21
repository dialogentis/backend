import asyncio
from datetime import UTC, datetime
from typing import Annotated, Literal, cast
from uuid import UUID

import aio_pika
from aio_pika.exceptions import AMQPException, ChannelNotFoundEntity
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select, tuple_

from dialogentis.auth.security import VerifiedUser
from dialogentis.config import Settings
from dialogentis.contracts import CursorPage
from dialogentis.datasets.export_routes import JobStatus
from dialogentis.dependencies import Database
from dialogentis.jobs.models import JOB_QUEUE_NAMES, Job, OutboxEvent
from dialogentis.models import Agent
from dialogentis.organizations import organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/jobs", tags=["jobs"])
type WorkerState = Literal["listening", "absent", "unknown"]
type QueuePhase = Literal["retry_wait", "dispatch_wait", "worker_wait"]


class JobBrokerState(BaseModel):
    status: Literal["available", "unavailable"] = "unavailable"
    normal_worker: WorkerState = "unknown"
    training_worker: WorkerState = "unknown"


class WorkspaceJob(BaseModel):
    job_id: UUID
    kind: str
    status: JobStatus
    queue: Literal["normal", "training"]
    cancel_requested: bool
    error_code: str | None
    created_at: datetime
    retry_at: datetime | None
    phase: QueuePhase | None


class WorkspaceJobs(CursorPage[WorkspaceJob]):
    active_agents: int
    queued: int
    running: int
    failed: int
    as_of: datetime
    broker: JobBrokerState


async def broker_state(config: Settings) -> JobBrokerState:
    result = JobBrokerState()
    if config.broker_url is None:
        return result
    try:
        async with (
            asyncio.timeout(1.5),
            await aio_pika.connect(config.broker_url.get_secret_value()) as connection,
        ):
            result.status = "available"

            async def worker(name: str) -> WorkerState:
                try:
                    async with connection.channel() as channel:
                        queue = await channel.declare_queue(name, passive=True)
                        return "listening" if queue.declaration_result.consumer_count else "absent"
                except ChannelNotFoundEntity:
                    return "absent"
                except AMQPException, OSError:
                    return "unknown"

            result.normal_worker, result.training_worker = await asyncio.gather(
                worker(JOB_QUEUE_NAMES["normal"]), worker(JOB_QUEUE_NAMES["training"])
            )
    except TimeoutError, AMQPException, OSError:
        pass
    return result


def job_view(job: Job, unpublished: datetime | None, now: datetime) -> WorkspaceJob:
    phase: QueuePhase | None = None
    if job.status == "queued":
        if job.next_retry_at is not None and job.next_retry_at > now:
            phase = "retry_wait"
        else:
            phase = "dispatch_wait" if unpublished is not None else "worker_wait"
    return WorkspaceJob(
        job_id=job.id,
        kind=job.kind,
        status=cast(JobStatus, job.status),
        queue=job.queue,
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        created_at=job.created_at,
        retry_at=job.next_retry_at if phase == "retry_wait" else None,
        phase=phase,
    )


@router.get("/queue", response_model=WorkspaceJobs)
async def workspace_jobs(
    org_id: UUID,
    request: Request,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: UUID | None = None,
    status: JobStatus | Literal["active"] | None = None,
):
    member = await organization_context(db, user, org_id)
    tenant = [Job.organization_id == org_id]
    if member.role == "viewer":
        tenant.append(Job.kind == "agent.training")
    elif member.role == "editor":
        tenant.append(Job.kind.in_(("agent.training", "dataset.export", "dataset.import")))
    totals = dict(
        (
            await db.execute(
                select(Job.status, func.count())
                .where(*tenant, Job.status.in_(("queued", "running", "failed")))
                .group_by(Job.status)
            )
        ).all()
    )
    count = await db.scalar(
        select(func.count())
        .select_from(Agent)
        .where(
            Agent.organization_id == org_id,
            Agent.status == "active",
        )
    )
    visible = [*tenant]
    if status == "active":
        visible.append(Job.status.in_(("queued", "running")))
    elif status is not None:
        visible.append(Job.status == status)
    if cursor is not None:
        boundary = await db.scalar(select(Job).where(*visible, Job.id == cursor))
        if boundary is None:
            raise HTTPException(404, "Страница очереди недоступна")
        visible.append(tuple_(Job.created_at, Job.id) < (boundary.created_at, cursor))
    unpublished = (
        select(func.min(OutboxEvent.available_at))
        .where(
            OutboxEvent.organization_id == org_id,
            OutboxEvent.job_id == Job.id,
            OutboxEvent.published_at.is_(None),
        )
        .correlate(Job)
        .scalar_subquery()
    )
    rows = (
        await db.execute(
            select(Job, unpublished)
            .where(*visible)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(limit + 1)
        )
    ).all()
    now = datetime.now(UTC)
    return WorkspaceJobs(
        active_agents=count or 0,
        queued=totals.get("queued", 0),
        running=totals.get("running", 0),
        failed=totals.get("failed", 0),
        as_of=now,
        broker=await broker_state(request.app.state.settings),
        items=[job_view(job, pending, now) for job, pending in rows[:limit]],
        next_cursor=str(rows[limit - 1][0].id) if len(rows) > limit else None,
    )
