import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import Context, TaskiqDepends, TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import ArtifactUnavailable, cleanup_expired, cleanup_orphans
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.datasets.export_worker import scrub_exports
from dialogentis.datasets.import_worker import scrub_imports
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.broker import create_broker
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.lifecycle import recover_tenant
from dialogentis.jobs.models import MaintenanceRun, OutboxEvent
from dialogentis.runtime.cleanup import cleanup_expired_conversations
from dialogentis.runtime.models import TestConversation

broker = create_broker("dialogentis.maintenance")


async def reconcile_once(
    factory: async_sessionmaker[AsyncSession],
    dispatch_factory: async_sessionmaker[AsyncSession],
    window: datetime,
    retryable_kinds: frozenset[str],
) -> bool:
    async with unit_of_work(factory) as guard:
        locked = await guard.scalar(select(func.pg_try_advisory_xact_lock(0x444C47454E54)))
        if not locked or await guard.get(MaintenanceRun, ("jobs.reconcile", window)):
            return False
        async with dispatch_factory() as db:
            organizations = (await db.scalars(select(OutboxEvent.organization_id).distinct())).all()
            conversation_organizations = (
                await db.scalars(select(TestConversation.organization_id).distinct())
            ).all()
            artifact_organizations = (
                await db.scalars(select(Artifact.organization_id).distinct())
            ).all()
        for organization_id in organizations:
            for _ in range(10):
                if await recover_tenant(factory, organization_id, retryable_kinds) < 100:
                    break
            async with unit_of_work(factory, organization_id=organization_id) as db:
                await scrub_exports(db, organization_id)
                await scrub_imports(db, organization_id)
        for organization_id in conversation_organizations:
            for _ in range(10):
                async with unit_of_work(factory, organization_id=organization_id) as db:
                    removed = await cleanup_expired_conversations(db, organization_id)
                if removed < 100:
                    break
        storage = ArtifactStorage(get_settings().artifact_root)
        filesystem_organizations = await asyncio.to_thread(storage.organizations)
        for organization_id in set(artifact_organizations) | set(filesystem_organizations):
            for _ in range(10):
                try:
                    async with unit_of_work(factory, organization_id=organization_id) as db:
                        removed = await cleanup_expired(db, storage, organization_id)
                except ArtifactUnavailable:
                    break
                if removed < 100:
                    break
            try:
                async with unit_of_work(factory, organization_id=organization_id) as db:
                    await cleanup_orphans(db, storage, organization_id)
            except ArtifactUnavailable:
                # shortcut: unknown tenants need review; the journal handles confirmed deletions.
                pass
        guard.add(MaintenanceRun(name="jobs.reconcile", window_start=window))
        await guard.execute(
            delete(MaintenanceRun).where(MaintenanceRun.window_start < window - timedelta(days=7))
        )
        return True


@broker.task(ack_type="manual", schedule=[{"cron": "* * * * *", "schedule_id": "jobs-reconcile"}])
async def reconcile(context: Context = TaskiqDepends()) -> None:
    factory = None
    dispatch_factory = None
    try:
        url = get_settings().database_url
        if url is None:
            raise ValueError("DATABASE_URL is required")
        factory = session_factory(url.get_secret_value())
        dispatch_factory = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        async with unit_of_work(factory) as db:
            window = await db.scalar(select(func.date_trunc("minute", func.now())))
            assert window is not None
        from dialogentis.jobs.tasks import HANDLERS

        retryable = frozenset(kind for kind, handler in HANDLERS.items() if handler.retryable)
        await reconcile_once(factory, dispatch_factory, window, retryable)
        await context.ack()
    except Exception:
        if broker.read_channel is not None:
            await broker.read_channel.close()
        raise
    finally:
        if dispatch_factory is not None:
            await dispatch_factory.kw["bind"].dispose()
        if factory is not None:
            await factory.kw["bind"].dispose()


scheduler = TaskiqScheduler(broker, [LabelScheduleSource(broker)])
