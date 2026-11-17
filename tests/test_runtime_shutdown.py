import asyncio
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.runtime.executor import RuntimeExecutor
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
