import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dialogentis.jobs.models import OutboxEvent

Publisher = Callable[[UUID, UUID], Awaitable[None]]


def dispatcher_factory(password_file: Path) -> async_sessionmaker[AsyncSession]:
    url = URL.create(
        "postgresql+psycopg",
        username="dialogentis_dispatcher",
        password=password_file.read_text().strip(),
        host="postgres",
        database="dialogentis",
    )
    return async_sessionmaker(create_async_engine(url, pool_pre_ping=True), expire_on_commit=False)


async def dispatch_once(factory: async_sessionmaker[AsyncSession], publish: Publisher) -> int:
    async with factory() as db, db.begin():
        events = (
            await db.scalars(
                select(OutboxEvent)
                .where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.available_at <= datetime.now(UTC),
                )
                .order_by(OutboxEvent.available_at, OutboxEvent.id)
                .limit(20)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for event in events:
            await publish(event.job_id, event.organization_id)
            event.published_at = datetime.now(UTC)
        return len(events)


async def main() -> None:
    from dialogentis.jobs.tasks import broker, run_job

    factory = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    try:
        await broker.startup()
        if broker.write_channel is None or not broker.write_channel.publisher_confirms:
            raise RuntimeError("RabbitMQ publisher confirms are required")

        async def publish(job_id: UUID, organization_id: UUID) -> None:
            await run_job.kiq(job_id, organization_id)

        while True:
            try:
                await dispatch_once(factory, publish)
            except Exception:
                logging.warning("Unable to dispatch jobs; retrying")
            await asyncio.sleep(1)
    finally:
        await broker.shutdown()
        await factory.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
