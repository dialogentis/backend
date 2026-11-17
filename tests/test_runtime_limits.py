import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from dialogentis.db import session_factory, unit_of_work
from dialogentis.runtime.limits import SLOT_BASE, SLOT_COUNT, run_limited
from tests.test_exports import app as app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_runtime_slots_are_shared_and_released_after_cancellation(app):
    assert app.state.settings.database_url is not None
    other_factory = session_factory(app.state.settings.database_url.get_secret_value())
    started = [asyncio.Event() for _ in range(SLOT_COUNT + 1)]
    release = asyncio.Event()

    async def work(index):
        started[index].set()
        await release.wait()
        return index

    tasks = []
    try:
        for index in range(SLOT_COUNT):
            factory = app.state.session_factory if index % 2 == 0 else other_factory
            tasks.append(asyncio.create_task(run_limited(factory, lambda i=index: work(i))))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started[:-1])), 10)
        waiting = asyncio.create_task(run_limited(other_factory, lambda: work(SLOT_COUNT)))
        tasks.append(waiting)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(started[-1].wait(), 0.3)

        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        await asyncio.wait_for(started[-1].wait(), 10)
        release.set()
        assert sorted(await asyncio.gather(*tasks[1:])) == list(range(1, SLOT_COUNT + 1))
        assert await asyncio.wait_for(run_limited(other_factory, lambda: work(0)), 10) == 0
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await other_factory.kw["bind"].dispose()


async def test_lost_runtime_slot_connection_cancels_work(app):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(run_limited(app.state.session_factory, work))
    try:
        await asyncio.wait_for(started.wait(), 10)
        async with unit_of_work(app.state.session_factory) as db:
            pid = await db.scalar(
                text(
                    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted "
                    "AND classid = :high AND objid = :low"
                ),
                {"high": SLOT_BASE >> 32, "low": SLOT_BASE & 0xFFFFFFFF},
            )
            assert pid is not None
            assert await db.scalar(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        with pytest.raises(DBAPIError):
            await asyncio.wait_for(task, 10)
        assert stopped.is_set()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
