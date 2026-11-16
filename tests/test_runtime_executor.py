import asyncio
import sys
from uuid import uuid4

import pytest

from dialogentis.runtime.core import RuntimeState
from dialogentis.runtime.executor import ExecutionError, RuntimeExecutor
from dialogentis.training.worker import run_child
from dialogentis.versions.prepare import canonical_hash
from tests.test_training_child import request_data


async def test_child_executes_real_signed_model_and_rejects_wrong_identity_or_snapshot():
    request, identity, snapshot, settings = request_data()

    async def progress(_):
        pass

    model, _ = await run_child(request, settings, progress)
    executor = RuntimeExecutor(timeout=20)
    first = await executor.execute(
        action="start",
        snapshot=snapshot,
        config_hash=canonical_hash(snapshot),
        identity=identity,
        model=model,
        settings=settings,
    )
    assert first.state.status == "waiting_input" and first.state.turn == 0
    second = await executor.execute(
        action="send",
        snapshot=snapshot,
        config_hash=canonical_hash(snapshot),
        identity=identity,
        model=model,
        settings=settings,
        state=first.state,
        text="👩🏽‍💻 доставьте в Москва номер 8",
    )
    assert second.state.turn == 1 and all("Москва" not in repr(item) for item in second.trace)
    with pytest.raises(ExecutionError, match="runtime_failed"):
        await executor.execute(
            action="start",
            snapshot=snapshot,
            config_hash="0" * 64,
            identity=identity,
            model=model,
            settings=settings,
        )
    from dataclasses import replace

    with pytest.raises(ExecutionError, match="runtime_failed"):
        await executor.execute(
            action="start",
            snapshot=snapshot,
            config_hash=canonical_hash(snapshot),
            identity=replace(identity, organization_id=uuid4()),
            model=model,
            settings=settings,
        )


async def test_timeout_kills_subprocess_and_does_not_leak_input(monkeypatch):
    _, identity, snapshot, settings = request_data()
    original = asyncio.create_subprocess_exec
    children = []

    async def slow(*args, **kwargs):
        process = await original(sys.executable, "-c", "import time; time.sleep(120)", **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("dialogentis.runtime.executor.asyncio.create_subprocess_exec", slow)
    with pytest.raises(ExecutionError, match="^runtime_timeout$") as failure:
        await RuntimeExecutor(timeout=0.3).execute(
            action="send",
            snapshot=snapshot,
            config_hash="a" * 64,
            identity=identity,
            model=b"model",
            settings=settings,
            state=RuntimeState(
                version_id=identity.version_id,
                flow_hash="a" * 64,
                node_id=uuid4(),
                status="waiting_input",
                turn=0,
                fallback_count=0,
            ),
            text="SECRET_TEST",
        )
    assert "SECRET_TEST" not in str(failure.value)
    assert children and children[0].returncode is not None


async def test_cancel_kills_child_and_limits_concurrency(monkeypatch):
    _, identity, snapshot, settings = request_data()
    original = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def slow(*args, **kwargs):
        process = await original(sys.executable, "-c", "import time; time.sleep(120)", **kwargs)
        children.append(process)
        started.set()
        return process

    monkeypatch.setattr("dialogentis.runtime.executor.asyncio.create_subprocess_exec", slow)
    executor = RuntimeExecutor(concurrency=1, timeout=10)

    def run():
        return executor.execute(
            action="start",
            snapshot=snapshot,
            config_hash="a" * 64,
            identity=identity,
            model=b"model",
            settings=settings,
        )

    first = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)
    second = asyncio.create_task(run())
    await asyncio.sleep(0.1)
    assert len(children) == 1
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert children[0].returncode is not None


async def test_rejects_invalid_message_before_starting_child(monkeypatch):
    _, identity, snapshot, settings = request_data()
    executor = RuntimeExecutor()
    state = RuntimeState(
        version_id=identity.version_id,
        flow_hash="a" * 64,
        node_id=uuid4(),
        status="waiting_input",
        turn=0,
        fallback_count=0,
    )

    async def no_spawn(*args, **kwargs):
        raise AssertionError("Invalid request must not start a process")

    monkeypatch.setattr("dialogentis.runtime.executor.asyncio.create_subprocess_exec", no_spawn)
    for text in ("", "x" * 2001, "\ud800"):
        with pytest.raises(ExecutionError, match="runtime_invalid_request"):
            await executor.execute(
                action="send",
                snapshot=snapshot,
                config_hash="a" * 64,
                identity=identity,
                model=b"model",
                settings=settings,
                state=state,
                text=text,
            )
