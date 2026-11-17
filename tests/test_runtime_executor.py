import asyncio
import sys
from uuid import UUID, uuid4

import pytest

from dialogentis.http_action import HTTP_OUTPUTS
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


async def test_http_bridge_validates_and_bounds_private_action_exchange():
    request, identity, snapshot, settings = request_data()

    async def progress(_):
        pass

    model, _ = await run_child(request, settings, progress)
    config = snapshot["config"]
    source_id, target_id, action_id, say_id = (str(uuid4()) for _ in range(4))
    config["slots"] = [
        {"id": source_id, "code": "order", "default": "private-order", "sensitive": True},
        {"id": target_id, "code": "status", "sensitive": True},
    ]
    flow = config["flow"]
    start, switch, _, end = flow["nodes"]
    action = {
        "id": action_id,
        "type": "http_action",
        "position": {"x": 0, "y": 0},
        "config": {
            "integration_id": str(uuid4()),
            "path": "/orders",
            "query": {"id": source_id},
            "response_mapping": {"status": target_id},
        },
    }
    say = {
        "id": say_id,
        "type": "say",
        "position": {"x": 0, "y": 0},
        "config": {"text": "{{status}}"},
    }
    flow["nodes"] += [action, say]
    flow["edges"][0]["target"] = action_id
    flow["edges"] += [
        {
            "id": str(uuid4()),
            "source": action_id,
            "source_handle": handle,
            "target": say_id if handle == "success" else end["id"],
        }
        for handle in HTTP_OUTPUTS
    ]
    flow["edges"].append(
        {"id": str(uuid4()), "source": say_id, "source_handle": "next", "target": switch["id"]}
    )
    seen = []

    async def handler(node_id, values):
        seen.append((node_id, values))
        return "success", {"status": "ready"}

    async def execute(callback, limit=20):
        return await RuntimeExecutor(timeout=limit).execute(
            action="start",
            snapshot=snapshot,
            config_hash=canonical_hash(snapshot),
            identity=identity,
            model=model,
            settings=settings,
            http_handler=callback,
        )

    result = await execute(handler)
    assert seen == [(UUID(action_id), {"order": "private-order"})]
    assert result.messages == ("ready",) and result.state.status == "waiting_input"
    assert "private-order" not in repr(result.trace) and "ready" not in repr(result.trace)

    async def invalid(_, __):
        return "success", {"order": "tampered"}

    with pytest.raises(ExecutionError, match="runtime_invalid_response"):
        await execute(invalid)

    async def slow(_, __):
        await asyncio.sleep(10)
        return "success", {}

    with pytest.raises(ExecutionError, match="runtime_timeout"):
        await execute(slow, 2)


async def test_http_bridge_rejects_action_not_in_snapshot(monkeypatch):
    _, identity, snapshot, settings = request_data()
    original = asyncio.create_subprocess_exec
    program = """
import json, struct, sys
request_size, model_size = struct.unpack('>II', sys.stdin.buffer.read(8))
sys.stdin.buffer.read(request_size + model_size)
payload = json.dumps({'kind': 'http_action',
                      'node_id': '00000000-0000-0000-0000-000000000001', 'values': {}}).encode()
sys.stdout.buffer.write(struct.pack('>I', len(payload)) + payload)
sys.stdout.buffer.flush()
sys.stdin.buffer.read()
"""

    async def fake_child(*args, **kwargs):
        return await original(sys.executable, "-c", program, **kwargs)

    async def handler(_, __):
        raise AssertionError("Unrecognized node must not reach handler")

    monkeypatch.setattr("dialogentis.runtime.executor.asyncio.create_subprocess_exec", fake_child)
    with pytest.raises(ExecutionError, match="runtime_invalid_response"):
        await RuntimeExecutor().execute(
            action="start",
            snapshot=snapshot,
            config_hash=canonical_hash(snapshot),
            identity=identity,
            model=b"model",
            settings=settings,
            http_handler=handler,
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
