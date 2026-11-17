"""Platform-wide slots for child-process runtime execution."""

import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

SLOT_BASE = 0x444C475254000000
SLOT_COUNT = 4


async def run_limited[T](
    factory: async_sessionmaker[AsyncSession], work: Callable[[], Awaitable[T]]
) -> T:
    async with factory() as session:
        task: asyncio.Task[T] | None = None
        monitor: asyncio.Task[None] | None = None
        try:
            while True:
                for slot in range(SLOT_COUNT):
                    if await session.scalar(
                        text("SELECT pg_try_advisory_lock(:key)"), {"key": SLOT_BASE + slot}
                    ):
                        break
                else:
                    await asyncio.sleep(0.1)
                    continue
                break

            async def check_connection() -> None:
                while True:
                    await asyncio.sleep(1)
                    async with asyncio.timeout(2):
                        await session.execute(text("SELECT 1"))

            async def run_work() -> T:
                return await work()

            task = asyncio.create_task(run_work())
            monitor = asyncio.create_task(check_connection())
            done, _ = await asyncio.wait((task, monitor), return_when=asyncio.FIRST_COMPLETED)
            if monitor in done:
                await monitor
                raise RuntimeError("Runtime slot monitor stopped")
            return await task
        finally:
            for running in (task, monitor):
                if running is not None and not running.done():
                    running.cancel()

            async def cleanup() -> None:
                await asyncio.gather(
                    *(item for item in (task, monitor) if item is not None), return_exceptions=True
                )
                await session.invalidate()

            closing = asyncio.create_task(cleanup())
            interrupted = False
            while True:
                try:
                    await asyncio.shield(closing)
                    break
                except asyncio.CancelledError:
                    if closing.cancelled():
                        raise
                    interrupted = True
            if interrupted:
                raise asyncio.CancelledError
