import asyncio
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from dialogentis.agents import empty_config
from dialogentis.config import Settings
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.model_store import ModelIdentity, unpack_model
from dialogentis.training.worker import run_child, verify_output
from tests.test_nlu_learned_entities import learned_dataset


def request_data():
    dataset, _ = learned_dataset()
    config = empty_config()
    config.update(dataset.model_dump(mode="json"))
    identity = ModelIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    snapshot = {"schema_version": 1, "config": config, "nlu_settings": asdict(Thresholds())}
    request = {
        "organization_id": str(identity.organization_id),
        "agent_id": str(identity.agent_id),
        "version_id": str(identity.version_id),
        "artifact_id": str(identity.artifact_id),
        "snapshot": snapshot,
    }
    settings = Settings(encryption_key=Fernet.generate_key().decode())
    return request, identity, snapshot, settings


async def test_real_child_trains_evaluates_and_packages_a_signed_model():
    request, identity, snapshot, settings = request_data()
    stages = []

    async def progress(stage):
        stages.append(stage)

    model, report = await run_child(request, settings, progress)
    assert stages == ["validation", "intent_training", "entity_training", "evaluation", "packaging"]
    verify_output(model, report, identity, snapshot, settings)
    adapter = unpack_model(model, identity, settings)
    assert adapter.predict("👩🏽‍💻 доставьте в Москва номер 8").entities.matches[0].text == "Москва"
    assert report["graph_validation"] == "not_checked"
    assert report["intents"]["test_count"] == 2


async def test_quality_thresholds_only_change_warnings():
    request, _, snapshot, settings = request_data()
    snapshot["config"]["quality_settings"] = {"intent_f1": 0.0, "entity_f1": 0.0}

    async def progress(stage):
        pass

    _, baseline = await run_child(request, settings, progress)
    snapshot["config"]["quality_settings"] = {"intent_f1": 1.0, "entity_f1": 1.0}
    _, strict = await run_child(request, settings, progress)
    assert baseline["intents"] == strict["intents"]
    assert baseline["entities"] == strict["entities"]
    assert "low_intent_quality" not in baseline["warnings"]
    assert "low_entity_quality" not in baseline["warnings"]
    for metric, warning in (
        (strict["intents"]["macro_f1"], "low_intent_quality"),
        (strict["entities"]["f1"], "low_entity_quality"),
    ):
        assert (warning in strict["warnings"]) == (metric is not None and metric < 1.0)


async def test_cancelling_training_stops_child_process(monkeypatch):
    request, _, _, settings = request_data()
    original = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def slow_process(*args, **kwargs):
        process = await original(args[0], "-m", "tests.training_slow_child", **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("dialogentis.training.worker.asyncio.create_subprocess_exec", slow_process)

    async def progress(stage):
        if stage == "validation":
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run_child(request, settings, progress))
    async with asyncio.timeout(10):
        await started.wait()
        assert children[0].returncode is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert children[0].returncode is not None


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc").exists(), reason="Requires Linux process status"
)
async def test_child_stops_if_worker_is_killed():
    program = """
import asyncio, os
from dialogentis.config import Settings
from dialogentis.training import worker
original = asyncio.create_subprocess_exec
async def spawn(*args, **kwargs):
    process = await original(args[0], '-m', 'tests.training_slow_child', **kwargs)
    print(process.pid, flush=True)
    return process
worker.asyncio.create_subprocess_exec = spawn
async def progress(stage):
    print('ready', flush=True)
    await asyncio.Event().wait()
settings = Settings(encryption_key=os.environ['TRAINING_TEST_KEY'])
asyncio.run(worker.run_child({}, settings, progress))
"""
    supervisor = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        program,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={**os.environ, "TRAINING_TEST_KEY": Fernet.generate_key().decode()},
    )
    child_pid = None
    child_stopped = False
    try:
        assert supervisor.stdout is not None
        async with asyncio.timeout(15):
            child_pid = int((await supervisor.stdout.readline()).decode().strip())
            assert (await supervisor.stdout.readline()).strip() == b"ready"
            supervisor.kill()
            await supervisor.wait()
            stat = Path(f"/proc/{child_pid}/stat")
            while True:
                try:
                    state = (await asyncio.to_thread(stat.read_text)).rsplit(")", 1)[1].strip()[0]
                except FileNotFoundError:
                    break
                if state == "Z":
                    break
                await asyncio.sleep(0.05)
            child_stopped = True
    finally:
        if supervisor.returncode is None:
            supervisor.kill()
            await supervisor.wait()
        if child_pid is not None and not child_stopped:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
