import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.config import get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.deletion_journal import read_markers
from dialogentis.deletion_worker import purge_tenant
from dialogentis.legacy_mail import has_untagged_invitation_mail


async def replay(
    factory: async_sessionmaker[AsyncSession], root: Path, encryption_key: str | None
) -> int:
    markers = await asyncio.to_thread(read_markers, root)
    if not markers:
        return 0
    if encryption_key is None:
        raise ValueError("Encryption key is required to validate legacy mail")
    async with unit_of_work(factory) as db:
        if await has_untagged_invitation_mail(db, encryption_key):
            raise ValueError("Legacy invitation mail must be drained before deletion replay")
    for organization_id in markers:
        async with unit_of_work(factory, organization_id=organization_id) as db:
            await purge_tenant(db, organization_id, restoring=True)
    return len(markers)


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
