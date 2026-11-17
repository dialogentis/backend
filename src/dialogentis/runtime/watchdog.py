"""Recover expired runtime reservations independently of the minute scheduler."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.config import get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.runtime.cleanup import recover_abandoned_turns
from dialogentis.runtime.models import TestConversation


async def recover_once(
    factory: async_sessionmaker[AsyncSession],
    dispatch_factory: async_sessionmaker[AsyncSession],
) -> int:
    async with asyncio.timeout(5), dispatch_factory() as db:
        organizations = (
            await db.scalars(select(TestConversation.organization_id).distinct())
        ).all()
    recovered = 0
    for organization_id in organizations:
        try:
            async with asyncio.timeout(5):
                for _ in range(10):
                    async with unit_of_work(factory, organization_id=organization_id) as db:
                        count = await recover_abandoned_turns(db, organization_id)
                    recovered += count
                    if count < 100:
                        break
        except Exception:
            logging.warning("Unable to recover runtime turns for an organization; retrying")
    return recovered


async def main() -> None:
    url = get_settings().database_url
    if url is None:
        raise ValueError("DATABASE_URL is required")
    factory = session_factory(url.get_secret_value())
    dispatch_factory = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    try:
        while True:
            try:
                await recover_once(factory, dispatch_factory)
            except Exception:
                logging.warning("Unable to discover runtime turns; retrying")
            await asyncio.sleep(5)
    finally:
        await dispatch_factory.kw["bind"].dispose()
        await factory.kw["bind"].dispose()


if __name__ == "__main__":
    asyncio.run(main())
