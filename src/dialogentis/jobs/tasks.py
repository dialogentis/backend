from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import Context, TaskiqDepends

from dialogentis.config import get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.broker import broker
from dialogentis.jobs.models import Job


async def process_job(
    factory: async_sessionmaker[AsyncSession], job_id: UUID, organization_id: UUID
) -> None:
    async with unit_of_work(factory, organization_id=organization_id) as db:
        job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job is None or job.status != "queued":
            return
        if job.cancel_requested:
            job.status = "cancelled"
        else:
            # No domain handlers are enabled yet; unknown kinds must not be retried forever.
            job.attempts += 1
            job.status = "failed"
            job.error_code = "unsupported_job_kind"
        job.finished_at = datetime.now(UTC)


@broker.task(ack_type="manual")
async def run_job(job_id: UUID, organization_id: UUID, context: Context = TaskiqDepends()) -> None:
    try:
        url = get_settings().database_url
        if url is None:
            raise ValueError("DATABASE_URL is required")
        factory = session_factory(url.get_secret_value())
        try:
            await process_job(factory, job_id, organization_id)
        finally:
            await factory.kw["bind"].dispose()
        await context.ack()
    except Exception:
        # Closing the consumer channel requeues the unacknowledged message.
        if broker.read_channel is not None:
            await broker.read_channel.close()
        raise
