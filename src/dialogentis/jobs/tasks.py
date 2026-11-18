import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import Context, TaskiqDepends

from dialogentis.config import get_settings
from dialogentis.datasets.export_worker import prepare_export
from dialogentis.datasets.import_worker import prepare_import
from dialogentis.db import session_factory, unit_of_work
from dialogentis.deletion_worker import prepare_deletion
from dialogentis.jobs.broker import JobBroker, broker
from dialogentis.jobs.lifecycle import (
    Apply,
    Claim,
    PermanentJobError,
    RetryableJobError,
    cancel,
    claim,
    complete,
    fail,
    heartbeat,
)
from dialogentis.models import Organization
from dialogentis.runtime.history_export import prepare_history_export

HEARTBEAT_INTERVAL = 10


Prepare = Callable[[async_sessionmaker[AsyncSession], Claim], Awaitable[Apply]]


@dataclass(frozen=True)
class Handler:
    prepare: Prepare
    timeout_seconds: float
    retryable: bool


HANDLERS: dict[str, Handler] = {
    "organization.deletion": Handler(prepare_deletion, timeout_seconds=120, retryable=True),
    "dataset.export": Handler(prepare_export, timeout_seconds=120, retryable=True),
    "dataset.import": Handler(prepare_import, timeout_seconds=120, retryable=True),
    "history.export": Handler(prepare_history_export, timeout_seconds=120, retryable=True),
}


class OrganizationUnavailable(Exception):
    pass


async def _check_active(
    db: AsyncSession, organization_id: UUID, *, exclusive: bool = False
) -> None:
    org = await db.scalar(
        select(Organization)
        .where(Organization.id == organization_id)
        .with_for_update(read=not exclusive)
    )
    if org is None or org.deletion_requested_at is not None:
        raise OrganizationUnavailable


async def _watch(factory: async_sessionmaker[AsyncSession], current: Claim) -> bool | None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        result = await heartbeat(factory, current)
        if result is not False:
            return result
        if current.kind != "organization.deletion":
            try:
                async with unit_of_work(factory, organization_id=current.organization_id) as db:
                    await _check_active(db, current.organization_id)
            except OrganizationUnavailable:
                return (
                    True if await cancel(factory, current.job_id, current.organization_id) else None
                )


async def _no_apply(db: AsyncSession, job) -> None:
    pass


async def process_job(
    factory: async_sessionmaker[AsyncSession],
    job_id: UUID,
    organization_id: UUID,
    handlers: Mapping[str, Handler] | None = None,
    queue: Literal["normal", "training"] = "normal",
) -> None:
    handlers = HANDLERS if handlers is None else handlers
    current = await claim(factory, job_id, organization_id, frozenset(handlers), queue)
    if current is None:
        return
    if current.kind != "organization.deletion":
        try:
            async with unit_of_work(factory, organization_id=organization_id) as db:
                await _check_active(db, organization_id)
        except OrganizationUnavailable:
            await fail(factory, current, error_code="organization_unavailable", retryable=False)
            return
    handler = handlers[current.kind]
    prepare = asyncio.ensure_future(handler.prepare(factory, current))
    watch = asyncio.create_task(_watch(factory, current))

    async def stop() -> None:
        for task in (prepare, watch):
            if not task.done():
                task.cancel()
        await asyncio.gather(prepare, watch, return_exceptions=True)

    try:
        async with asyncio.timeout(handler.timeout_seconds):
            done, _ = await asyncio.wait((prepare, watch), return_when=asyncio.FIRST_COMPLETED)
            if watch in done:
                cancelled = watch.result()
                await stop()
                if cancelled:
                    await complete(factory, current, _no_apply)
                return
            apply = prepare.result()
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)

            async def guarded_apply(db: AsyncSession, job) -> None:
                if current.kind != "organization.deletion":
                    await _check_active(db, current.organization_id, exclusive=True)
                await apply(db, job)

            await complete(factory, current, guarded_apply)
    except TimeoutError:
        await stop()
        await fail(factory, current, error_code="timeout", retryable=handler.retryable)
    except RetryableJobError:
        await stop()
        await fail(factory, current, error_code="temporary_error", retryable=handler.retryable)
    except PermanentJobError as exc:
        await stop()
        await fail(factory, current, error_code=exc.code, retryable=False)
    except SQLAlchemyError:
        await stop()
        await fail(factory, current, error_code="database_error", retryable=handler.retryable)
    except OrganizationUnavailable:
        await stop()
        await fail(factory, current, error_code="organization_unavailable", retryable=False)
    except Exception:
        await stop()
        await fail(factory, current, error_code="handler_error", retryable=False)
    finally:
        await stop()


@broker.task(ack_type="manual")
async def run_job(job_id: UUID, organization_id: UUID, context: Context = TaskiqDepends()) -> None:
    await _run(broker, job_id, organization_id, context, HANDLERS, "normal")


async def _run(
    job_broker: JobBroker,
    job_id: UUID,
    organization_id: UUID,
    context: Context,
    handlers: Mapping[str, Handler],
    queue: Literal["normal", "training"],
) -> None:
    try:
        url = get_settings().database_url
        if url is None:
            raise ValueError("DATABASE_URL is required")
        factory = session_factory(url.get_secret_value())
        try:
            if queue == "training":
                from dialogentis.training.serial import run_serialized

                await run_serialized(
                    factory, lambda: process_job(factory, job_id, organization_id, handlers, queue)
                )
            else:
                await process_job(factory, job_id, organization_id, handlers, queue)
        finally:
            await factory.kw["bind"].dispose()
        await context.ack()
    except Exception:
        # Closing the consumer channel requeues the unacknowledged message.
        if job_broker.read_channel is not None:
            await job_broker.read_channel.close()
        raise
