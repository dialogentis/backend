"""Reconcile an offline database snapshot before starting the application."""

import asyncio
from pathlib import Path

from sqlalchemy import select, union, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import cleanup_expired
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.lifecycle import recover_tenant
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.jobs.tasks import HANDLERS
from dialogentis.jobs.training_tasks import HANDLERS as TRAINING_HANDLERS
from dialogentis.models import Organization
from dialogentis.runtime.cleanup import cleanup_expired_receipts
from dialogentis.runtime.history import cleanup_history
from dialogentis.runtime.history_deletion import expire_history_exports
from dialogentis.runtime.history_models import ConversationHistory
from dialogentis.runtime.receipt_models import RuntimeReceipt


async def prepare_restore(
    factory: async_sessionmaker[AsyncSession],
    discovery: async_sessionmaker[AsyncSession],
    storage: ArtifactStorage,
) -> None:
    if not storage.available():
        raise ValueError("Restored artifact volume is unavailable")
    async with discovery() as db:
        privacy_organizations = (
            await db.scalars(
                union(
                    select(ConversationHistory.organization_id),
                    select(RuntimeReceipt.organization_id),
                    select(Artifact.organization_id),
                )
            )
        ).all()
        job_organizations = (await db.scalars(select(OutboxEvent.organization_id).distinct())).all()

    for org_id in privacy_organizations:
        async with unit_of_work(factory, organization_id=org_id) as db:
            org = await db.get(Organization, org_id, with_for_update=True)
            if org is None or org.deletion_requested_at is not None:
                continue
            await cleanup_history(db, org)
            while await cleanup_expired_receipts(db, org_id) == 100:
                pass
            await expire_history_exports(db, org_id)
            while await cleanup_expired(db, storage, org_id) == 100:
                pass

    retryable = frozenset(
        kind for kind, handler in (HANDLERS | TRAINING_HANDLERS).items() if handler.retryable
    )
    for org_id in job_organizations:
        async with unit_of_work(factory, organization_id=org_id) as db:
            # The broker is empty on a clean restore. Republish only the latest event per job.
            latest_event = (
                select(OutboxEvent.id)
                .where(OutboxEvent.organization_id == org_id, OutboxEvent.job_id == Job.id)
                .order_by(OutboxEvent.available_at.desc(), OutboxEvent.id.desc())
                .limit(1)
                .correlate(Job)
                .scalar_subquery()
            )
            queued_events = select(latest_event).where(
                Job.organization_id == org_id, Job.status == "queued"
            )
            await db.execute(
                update(OutboxEvent)
                .where(OutboxEvent.organization_id == org_id, OutboxEvent.id.in_(queued_events))
                .values(published_at=None)
            )
            await db.execute(
                update(Job)
                .where(Job.organization_id == org_id, Job.status == "running")
                .values(heartbeat_at=None)
            )
        while await recover_tenant(factory, org_id, retryable) == 100:
            pass


async def main() -> None:
    config = get_settings()
    if config.database_url is None:
        raise ValueError("DATABASE_URL is required")
    factory = session_factory(config.database_url.get_secret_value())
    discovery = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    try:
        await prepare_restore(factory, discovery, ArtifactStorage(config.artifact_root))
    finally:
        await discovery.kw["bind"].dispose()
        await factory.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
