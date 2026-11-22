"""Measure the Dialogentis HTTP runtime and cabinet against a seeded Compose stack."""

import argparse
import asyncio
import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

import httpx

from benchmarks.runtime.seed import ANSWER, PROMPT, check, login, train

RUNTIME = "/api/v1/runtime"


@dataclass
class Samples:
    latency: list[float] = field(default_factory=list)
    errors: Counter[str] = field(default_factory=Counter)

    def report(self) -> dict:
        times = sorted(self.latency)
        return {
            "succeeded": len(times),
            "errors": dict(sorted(self.errors.items())),
            **{
                f"p{percent}_ms": round(times[math.ceil(len(times) * percent / 100) - 1] * 1000, 1)
                if times
                else None
                for percent in (50, 95, 99)
            },
        }


async def start(client: httpx.AsyncClient, agent: dict) -> tuple[str, dict]:
    auth = {"Authorization": f"Bearer {agent['key']}"}
    started = check(
        await client.post(
            f"{RUNTIME}/agents/{agent['id']}/conversations",
            headers={**auth, "Idempotency-Key": uuid4().hex},
        ),
        201,
    )
    if started["status"] != "waiting_input":
        raise RuntimeError("The published agent did not wait for input")
    path = f"{RUNTIME}/conversations/{started['conversation_id']}"
    return path, {**auth, "X-Conversation-Token": started["conversation_token"]}


async def send(client: httpx.AsyncClient, conversation: tuple[str, dict], samples: Samples):
    path, auth = conversation
    began = perf_counter()
    try:
        response = await client.post(
            f"{path}/messages",
            headers={**auth, "Idempotency-Key": uuid4().hex},
            json={"client_message_id": str(uuid4()), "text": PROMPT},
        )
        elapsed = perf_counter() - began
        if response.status_code != 200:
            samples.errors[f"http_{response.status_code}"] += 1
            return
        payload = response.json()
        if (
            payload["status"] == "waiting_input"
            and [item["text"] for item in payload["messages"]] == [ANSWER]
            and not payload["replayed"]
        ):
            samples.latency.append(elapsed)
        else:
            samples.errors["unexpected_reply"] += 1
    except httpx.RequestError as exc:
        samples.errors[type(exc).__name__] += 1
    except ValueError, KeyError, TypeError:
        samples.errors["invalid_response"] += 1


async def traffic(client, conversations, rate, begin, duration, samples):
    clock = asyncio.get_running_loop()
    wall_offset = time() - clock.time()
    scheduled = duration * rate
    pending = [None] * len(conversations)
    async with asyncio.TaskGroup() as tasks:
        for index in range(scheduled):
            when = begin + index / rate
            await asyncio.sleep(max(0, when - clock.time()))
            if abs(time() - clock.time() - wall_offset) >= 5 or clock.time() - when >= 5:
                raise RuntimeError("Measurement interrupted by a clock jump or host suspension")
            dialog = index % len(conversations)
            previous = pending[dialog]
            if clock.time() >= when + 1 / rate:
                samples.errors["late_schedule"] += 1
            elif previous is not None and not previous.done():
                samples.errors["dialog_busy"] += 1
            else:
                pending[dialog] = tasks.create_task(send(client, conversations[dialog], samples))
    assert len(samples.latency) + samples.errors.total() == scheduled


async def cabinet(client, agents, begin, duration, samples):
    clock = asyncio.get_running_loop()
    for index in range(0, duration, 2):
        await asyncio.sleep(max(0, begin + index - clock.time()))
        agent = agents[(index // 2) % len(agents)]
        for url in (agent["path"].rsplit("/", 1)[0], f"{agent['path']}/versions"):
            began = perf_counter()
            try:
                response = await client.get(url, params={"limit": 100})
                if response.status_code == 200:
                    samples.latency.append(perf_counter() - began)
                else:
                    samples.errors[f"http_{response.status_code}"] += 1
            except httpx.RequestError as exc:
                samples.errors[type(exc).__name__] += 1


async def background_training(client, agent, stop, windows):
    clock = asyncio.get_running_loop()
    while not stop.is_set():
        version = await train(client, agent["path"], agent["revision"])
        path = f"{agent['path']}/versions/{version}"
        first_running = None
        last_running = None
        async with asyncio.timeout(900):
            while True:
                job = check(await client.get(path), 200)
                if job["status"] == "ready":
                    if first_running is not None and last_running is not None:
                        windows.append((first_running, last_running))
                    break
                if job["status"] != "building":
                    raise RuntimeError(f"Concurrent training ended with {job['status']}")
                if job["job_status"] == "running":
                    last_running = clock.time()
                    if first_running is None:
                        first_running = last_running
                await asyncio.sleep(0.25)


async def profile(public, admin, agents, training, rate, warm, duration, dialogs):
    label = "with_training" if training else "baseline"
    conversations = [await start(public, agent) for _ in range(dialogs) for agent in agents]
    stop = asyncio.Event()
    windows = []
    warm_samples, messages, gets = Samples(), Samples(), Samples()
    close_errors = Counter()
    clock = asyncio.get_running_loop()
    try:
        async with asyncio.TaskGroup() as tasks:
            if training is not None:
                tasks.create_task(background_training(admin, training, stop, windows))
            begin = clock.time()
            await traffic(public, conversations, rate, begin, warm, warm_samples)
            await asyncio.sleep(max(0, begin + warm - clock.time()))
            measured_begin = clock.time()
            measurement_timestamp = datetime.now(UTC).isoformat()
            print(f"Measuring {label} at {measurement_timestamp}", flush=True)
            try:
                tasks.create_task(cabinet(admin, agents, measured_begin, duration, gets))
                await traffic(public, conversations, rate, measured_begin, duration, messages)
            finally:
                stop.set()
    finally:
        for path, auth in conversations:
            try:
                response = await public.post(f"{path}/close", headers=auth)
                if response.status_code != 204:
                    close_errors[f"http_{response.status_code}"] += 1
            except httpx.RequestError as exc:
                close_errors[type(exc).__name__] += 1
    measured_end = measured_begin + duration
    overlap = sum(
        max(0, min(end, measured_end) - max(begin, measured_begin)) for begin, end in windows
    )
    overtime = sum(int(end > measured_begin and begin < measured_end) for begin, end in windows)
    report = {
        "runtime": messages.report(),
        "cabinet_gets": gets.report(),
        "warmup": warm_samples.report(),
        "measurement_started_at": measurement_timestamp,
        "warmup_seconds": warm,
        "measurement_seconds": duration,
        "messages_per_second": rate,
        "active_dialogs": len(conversations),
        "observed_training_jobs_during_measurement": overtime,
        "observed_running_seconds_lower_bound": round(overlap, 1),
        "dialog_close_errors": dict(close_errors),
    }
    print(json.dumps({label: report}), flush=True)
    return report


def near_limit_draft(cfg: dict, limit: int) -> dict:
    def encoding(data):
        return json.dumps(data, ensure_ascii=False, allow_nan=False).encode()

    draft = deepcopy(cfg)
    size = len(encoding(draft))
    template = draft["examples"][0]
    for index in range(13000 - len(draft["examples"])):
        row = {**template, "id": str(uuid4()), "spans": [], "text": f"проверка {index} "}
        remaining = limit - size - len(encoding(row)) - 2
        padding = min(1900, remaining)
        if padding <= 0:
            break
        row["text"] += "x" * padding
        size += len(encoding(row)) + 2
        draft["examples"].append(row)
    assert limit - 512 <= size == len(encoding(draft)) <= limit
    return draft


async def measure_saves(admin, agent, repeats):
    path = agent["path"]
    current = await admin.get(f"{path}/draft")
    original = check(current, 200)["config"]
    etag = current.headers["ETag"]
    bodies = {
        label: json.dumps(
            {"schema_version": 1, "config": near_limit_draft(original, limit - 64)},
            ensure_ascii=False,
        ).encode()
        for label, limit in (("1_mib", 1024**2), ("10_mib", 10 * 1024**2))
    }
    lease = check(await admin.post(f"{path}/edit-lease", json={"tab_id": str(uuid4())}), 200)
    headers = {
        "X-Edit-Lease": lease["lease_token"],
        "X-Edit-Fence": str(lease["fencing_counter"]),
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    results = {}
    try:
        for label, body in bodies.items():
            samples = Samples()
            for _ in range(repeats):
                check(await admin.post(f"{path}/edit-lease/renew", headers=headers), 200)
                began = perf_counter()
                response = await admin.put(
                    f"{path}/draft", headers={**headers, "If-Match": etag}, content=body
                )
                elapsed = perf_counter() - began
                if response.status_code != 200:
                    samples.errors[f"http_{response.status_code}"] += 1
                    continue
                etag = response.headers["ETag"]
                samples.latency.append(elapsed)
            results[label] = {"request_bytes": len(body), **samples.report()}
    finally:
        check(
            await admin.put(
                f"{path}/draft",
                headers={**headers, "If-Match": etag},
                content=json.dumps({"schema_version": 1, "config": original}).encode(),
            ),
            200,
        )
        check(await admin.delete(f"{path}/edit-lease", headers=headers), 204)
    return results


async def main(args):
    inventory = json.loads(args.inventory.read_text())
    url, smoke, agents = inventory["url"], inventory["smoke"], inventory["agents"]
    active = list({agent["organization"]: agent for agent in reversed(agents)}.values())
    cold = [agent for agent in agents if agent not in active]
    assert (len(agents), len(active)) == ((2, 1) if smoke else (100, 10))
    assert len({agent["id"] for agent in agents}) == len(agents)
    rate, warm, duration, dialogs = (2, 3, 10, 4) if smoke else (10, 120, 900, 10)
    report = {"smoke": smoke, "models": len(agents), "organizations": len(active)}

    def write_report():
        temporary = args.report.with_name(f"{args.report.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    async with (
        httpx.AsyncClient(base_url=url, timeout=30) as public,
        httpx.AsyncClient(base_url=url, headers={"Origin": url}, timeout=45) as admin,
    ):
        await login(admin, inventory["credentials"])
        cold_samples = Samples()
        for agent in cold:
            began = perf_counter()
            try:
                path, auth = await start(public, agent)
                cold_samples.latency.append(perf_counter() - began)
                check(await public.post(f"{path}/close", headers=auth), 204)
            except (RuntimeError, httpx.RequestError) as exc:
                cold_samples.errors[
                    str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                ] += 1
        report["cold_model_start"] = cold_samples.report()
        write_report()
        print(json.dumps({"cold_model_start": report["cold_model_start"]}), flush=True)
        report["baseline"] = await profile(
            public, admin, active, None, rate, warm, duration, dialogs
        )
        write_report()
        report["with_training"] = await profile(
            public, admin, active, cold[0], rate, warm, duration, dialogs
        )
        write_report()
        report["draft_saves"] = await measure_saves(admin, active[0], 1 if smoke else 20)
    write_report()
    print(json.dumps({"draft_saves": report["draft_saves"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("report", type=Path)
    asyncio.run(main(parser.parse_args()))
