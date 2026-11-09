"""A database-backed platform-wide training slot."""

import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

TRAINING_LOCK = 0x444C47545241494E


async def run_serialized(
    factory: async_sessionmaker[AsyncSession], work: Callable[[], Awaitable[None]]
) -> None:
    async with factory() as session:
        task: asyncio.Task | None = None
        monitor: asyncio.Task | None = None
        try:
            while not await session.scalar(  # noqa: ASYNC110
                text("SELECT pg_try_advisory_lock(:key)"), {"key": TRAINING_LOCK}
            ):
                await asyncio.sleep(0.25)

            async def check_connection() -> None:
                while True:
                    await asyncio.sleep(1)
                    async with asyncio.timeout(2):
                        await session.execute(text("SELECT 1"))

            async def run_work() -> None:
                await work()

            task = asyncio.create_task(run_work())
            monitor = asyncio.create_task(check_connection())
            done, _ = await asyncio.wait((task, monitor), return_when=asyncio.FIRST_COMPLETED)
            if monitor in done:
                await monitor
                raise RuntimeError("Training lock monitor stopped")
            await task
        finally:
            for running in (task, monitor):
                if running is not None and not running.done():
                    running.cancel()

            async def cleanup() -> None:
                await asyncio.gather(
                    *(item for item in (task, monitor) if item is not None), return_exceptions=True
                )
                # Closing rather than pooling this connection also releases uncertain locks.
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
