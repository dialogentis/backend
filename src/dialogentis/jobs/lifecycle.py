import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.db import unit_of_work
from dialogentis.jobs.models import Job, OutboxEvent

MAX_ATTEMPTS = 4
MAX_DELIVERY_RECOVERIES = 3
DELIVERY_TIMEOUT = timedelta(minutes=5)
HEARTBEAT_TIMEOUT = timedelta(seconds=90)
Apply = Callable[[AsyncSession, Job], Awaitable[None]]


class RetryableJobError(Exception):
    """A temporary error whose details must not be persisted in job metadata."""


class PermanentJobError(Exception):
    def __init__(self, code: str):
        self.code = code


@dataclass(frozen=True)
class Claim:
    job_id: UUID
    organization_id: UUID
    token: UUID
    kind: str


async def database_now(db: AsyncSession) -> datetime:
    now = await db.scalar(select(func.now()))
    assert now is not None
    return now


def retry_delay(attempt: int) -> timedelta:
    base = 5 * 2 ** max(0, attempt - 1)
    return timedelta(seconds=base + random.uniform(0, base * 0.2))


def finish(job: Job, status: str, now: datetime, error_code: str | None = None) -> None:
    job.status, job.finished_at, job.error_code = status, now, error_code
    job.execution_token = None
    job.next_retry_at = None


def schedule_retry(db: AsyncSession, job: Job, now: datetime, delay: timedelta) -> None:
    job.status = "queued"
    job.execution_token = None
    job.next_retry_at = now + delay
    db.add(
        OutboxEvent(
            organization_id=job.organization_id,
            job_id=job.id,
            queue=job.queue,
            available_at=job.next_retry_at,
        )
    )


async def claim(
    factory: async_sessionmaker[AsyncSession],
    job_id: UUID,
    organization_id: UUID,
    supported_kinds: frozenset[str],
    queue: Literal["normal", "training"] = "normal",
) -> Claim | None:
    async with unit_of_work(factory, organization_id=organization_id) as db:
        job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job is None or job.status != "queued" or job.queue != queue:
            return None
        now = await database_now(db)
        if job.cancel_requested:
            finish(job, "cancelled", now)
            return None
        if job.next_retry_at is not None and job.next_retry_at > now:
            return None
        if job.kind not in supported_kinds:
            job.attempts += 1
            finish(job, "failed", now, "unsupported_job_kind")
            return None
        job.attempts += 1
        job.status = "running"
        token = uuid4()
        job.execution_token = token
        job.started_at = job.heartbeat_at = now
        job.next_retry_at = None
        job.error_code = None
        return Claim(job.id, organization_id, token, job.kind)


async def heartbeat(factory: async_sessionmaker[AsyncSession], current: Claim) -> bool | None:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        result = await db.execute(
            update(Job)
            .where(
                Job.id == current.job_id,
                Job.status == "running",
                Job.execution_token == current.token,
            )
            .values(heartbeat_at=func.now())
            .returning(Job.cancel_requested)
        )
        return result.scalar_one_or_none()


async def complete(factory: async_sessionmaker[AsyncSession], current: Claim, apply: Apply) -> bool:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        job = await db.scalar(select(Job).where(Job.id == current.job_id).with_for_update())
        if job is None or job.status != "running" or job.execution_token != current.token:
            return False
        now = await database_now(db)
        if job.cancel_requested:
            finish(job, "cancelled", now)
        else:
            await apply(db, job)
            finish(job, "succeeded", now)
        return True


async def fail(
    factory: async_sessionmaker[AsyncSession], current: Claim, *, error_code: str, retryable: bool
) -> bool:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        job = await db.scalar(select(Job).where(Job.id == current.job_id).with_for_update())
        if job is None or job.status != "running" or job.execution_token != current.token:
            return False
        now = await database_now(db)
        if job.cancel_requested:
            finish(job, "cancelled", now)
        elif retryable and job.attempts < MAX_ATTEMPTS:
            job.error_code = error_code
            schedule_retry(db, job, now, retry_delay(job.attempts))
        else:
            finish(job, "failed", now, error_code)
        return True


async def cancel(
    factory: async_sessionmaker[AsyncSession], job_id: UUID, organization_id: UUID
) -> bool:
    async with unit_of_work(factory, organization_id=organization_id) as db:
        job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if (
            job is None
            or job.kind == "organization.deletion"
            or job.status not in {"queued", "running"}
        ):
            return False
        job.cancel_requested = True
        if job.status == "queued":
            finish(job, "cancelled", await database_now(db))
        return True


async def recover_tenant(
    factory: async_sessionmaker[AsyncSession],
    organization_id: UUID,
    retryable_kinds: frozenset[str],
) -> int:
    async with unit_of_work(factory, organization_id=organization_id) as db:
        now = await database_now(db)
        pending = exists(
            select(OutboxEvent.id).where(
                OutboxEvent.job_id == Job.id, OutboxEvent.published_at.is_(None)
            )
        )
        recent = exists(
            select(OutboxEvent.id).where(
                OutboxEvent.job_id == Job.id, OutboxEvent.published_at > now - DELIVERY_TIMEOUT
            )
        )
        published = exists(
            select(OutboxEvent.id).where(
                OutboxEvent.job_id == Job.id, OutboxEvent.published_at.is_not(None)
            )
        )
        stale_running = and_(
            Job.status == "running",
            or_(Job.heartbeat_at.is_(None), Job.heartbeat_at <= now - HEARTBEAT_TIMEOUT),
        )
        stale_delivery = and_(
            Job.status == "queued",
            or_(Job.next_retry_at.is_(None), Job.next_retry_at <= now),
            published,
            ~pending,
            ~recent,
        )
        jobs = (
            await db.scalars(
                select(Job)
                .where(or_(stale_running, stale_delivery))
                .order_by(Job.created_at, Job.id)
                .limit(100)
                .with_for_update(skip_locked=True)
            )
        ).all()
        recovered = 0
        for job in jobs:
            if job.status == "running":
                if job.cancel_requested:
                    finish(job, "cancelled", now)
                elif job.kind in retryable_kinds and job.attempts < MAX_ATTEMPTS:
                    job.error_code = "worker_lost"
                    schedule_retry(db, job, now, retry_delay(job.attempts))
                else:
                    finish(job, "failed", now, "worker_lost")
                recovered += 1
                continue
            if job.delivery_recoveries >= MAX_DELIVERY_RECOVERIES:
                finish(job, "failed", now, "delivery_lost")
            else:
                job.delivery_recoveries += 1
                job.error_code = "delivery_lost"
                schedule_retry(db, job, now, retry_delay(job.delivery_recoveries))
            recovered += 1
        return recovered
