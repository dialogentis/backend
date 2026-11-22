import asyncio
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.limits import SLOT_COUNT
from dialogentis.runtime.test_chat import ChatResponse
from dialogentis.runtime.test_chat import TestChatService as ChatService


async def test_lifespan_calls_chat_shutdown():
    app = create_app(Settings(_env_file=None))
    closed = asyncio.Event()

    class Chat:
        async def shutdown(self):
            closed.set()

    app.state.test_chat_service = Chat()
    async with app.router.lifespan_context(app):
        assert not closed.is_set()
    assert closed.is_set()


async def test_shutdown_waits_for_detached_turn_and_rejects_new_turns(monkeypatch):
    service = ChatService(
        cast(async_sessionmaker[AsyncSession], None), Settings(), RuntimeExecutor()
    )
    started, release = asyncio.Event(), asyncio.Event()

    async def work(*args):
        started.set()
        await release.wait()
        return ChatResponse(uuid4(), 2, "waiting_input", ("ответ",), ())

    monkeypatch.setattr(service, "_send", work)
    ids = [uuid4() for _ in range(4)]
    caller = asyncio.create_task(service.send(*ids[:3], 1, "текст", "token", ids[3], "key"))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    shutdown = asyncio.create_task(service.shutdown(grace_period=1))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as rejected:
        await service.send(*ids[:3], 1, "текст", "token", ids[3], "key")
    assert rejected.value.status_code == 503 and not shutdown.done()
    with pytest.raises(HTTPException) as rejected_start:
        await service.start(*ids)
    assert rejected_start.value.status_code == 503
    release.set()
    await shutdown
    assert not service._pending


async def test_shutdown_cancels_turn_after_timeout(monkeypatch):
    service = ChatService(
        cast(async_sessionmaker[AsyncSession], None), Settings(), RuntimeExecutor()
    )
    started, stopped = asyncio.Event(), asyncio.Event()

    async def work(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(service, "_send", work)
    ids = [uuid4() for _ in range(4)]
    caller = asyncio.create_task(service.send(*ids[:3], 1, "текст", "token", ids[3], "key"))
    await started.wait()
    await service.shutdown(grace_period=0.01)
    assert stopped.is_set() and not service._pending
    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_runtime_admission_limits_waiters_before_database_slot_polling(monkeypatch):
    service = ChatService(
        cast(async_sessionmaker[AsyncSession], None), Settings(), RuntimeExecutor()
    )
    release = asyncio.Event()
    first_wave = asyncio.Event()
    attempted = 0
    reply = ChatResponse(uuid4(), 1, "waiting_input", (), ())

    async def poll_slot(*args):
        nonlocal attempted
        attempted += 1
        if attempted == SLOT_COUNT:
            first_wave.set()
        await release.wait()
        return reply

    monkeypatch.setattr("dialogentis.runtime.test_chat.run_limited", poll_slot)
    ids = [uuid4() for _ in range(4)]
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(service.start(*ids)) for _ in range(SLOT_COUNT * 4)]
        try:
            await asyncio.wait_for(first_wave.wait(), 1)
            assert attempted == SLOT_COUNT
        finally:
            release.set()
    assert [task.result() for task in tasks] == [reply] * len(tasks)
    assert attempted == len(tasks)
    await service.shutdown()
