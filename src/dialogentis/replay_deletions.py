import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import cleanup_expired
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.deletion_journal import read_history_markers, read_history_policies, read_markers
from dialogentis.deletion_worker import purge_tenant
from dialogentis.legacy_mail import has_untagged_invitation_mail
from dialogentis.models import Organization
from dialogentis.runtime.history import apply_history_policy
from dialogentis.runtime.history_deletion import apply_history_marker, expire_history_exports


async def _remove_expired(
    db: AsyncSession, storage: ArtifactStorage, organization_id: UUID
) -> None:
    while await cleanup_expired(db, storage, organization_id) == 100:
        pass


async def replay(
    factory: async_sessionmaker[AsyncSession], root: Path, encryption_key: str | None
) -> int:
    markers = await asyncio.to_thread(read_markers, root)
    history = await asyncio.to_thread(read_history_markers, root)
    policies = await asyncio.to_thread(read_history_policies, root)
    if markers:
        if encryption_key is None:
            raise ValueError("Encryption key is required to validate legacy mail")
        async with unit_of_work(factory) as db:
            if await has_untagged_invitation_mail(db, encryption_key):
                raise ValueError("Legacy invitation mail must be drained before deletion replay")
    for organization_id in markers:
        async with unit_of_work(factory, organization_id=organization_id) as db:
            await purge_tenant(db, organization_id, restoring=True)
    storage = ArtifactStorage(get_settings().artifact_root)
    # The policy journal uses the saved revision, before replay of jobs can advance it.
    for policy in sorted(policies, key=lambda item: item.revision):
        async with unit_of_work(factory, organization_id=policy.organization_id) as db:
            org = await db.get(Organization, policy.organization_id, with_for_update=True)
            if (
                org is None
                or org.deletion_requested_at is not None
                or org.settings_revision >= policy.revision
            ):
                continue
            org.retention_days = policy.retention_days
            org.store_message_text = policy.store_message_text
            org.settings_revision = policy.revision
            await apply_history_policy(
                db, org, retention_cutoff=policy.retention_cutoff, purge_text=policy.purge_text
            )
            await expire_history_exports(db, org.id)
            await _remove_expired(db, storage, org.id)
    for marker in sorted(history, key=lambda item: item.requested_at):
        async with unit_of_work(factory, organization_id=marker.organization_id) as db:
            org = await db.get(Organization, marker.organization_id, with_for_update=True)
            if org is None or org.deletion_requested_at is not None:
                continue
            if any(await apply_history_marker(db, org, marker)):
                await _remove_expired(db, storage, org.id)
    return len(markers) + len(history) + len(policies)


async def main() -> None:
    settings = get_settings()
    if settings.database_url is None:
        raise ValueError("DATABASE_URL is required")
    factory = session_factory(settings.database_url.get_secret_value())
    try:
        key = settings.encryption_key.get_secret_value() if settings.encryption_key else None
        await replay(factory, settings.deletion_journal_root, key)
    finally:
        await factory.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
