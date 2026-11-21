"""Create the published load fixture through the public Dialogentis API."""

import argparse
import asyncio
import json
import os
import re
import secrets
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import httpx

from benchmarks.nlu.demo import demo_config, load_demo_fixtures

PROMPT = "где сейчас мой заказ"
ANSWER = "Запрос принят."
VERSION_TIMEOUT = 900


def check(response: httpx.Response, expected: int) -> dict:
    if response.status_code != expected:
        # Headers, credentials and request content never enter the benchmark log.
        raise RuntimeError(f"Unexpected HTTP {response.status_code} ({response.request.method})")
    return response.json() if response.content else {}


def config() -> dict:
    fixture = deepcopy(load_demo_fixtures()["order_support"])
    for intent in fixture["intents"]:
        phrases = intent["train"]
        intent["train"] = [
            f"{phrases[index % len(phrases)]} · запрос {index + 1}" if index else phrases[0]
            for index in range(100)
        ]
    cfg = demo_config("order_support", fixture)
    flow = cfg["flow"]
    switch = next(node["id"] for node in flow["nodes"] if node["type"] == "switch_intent")
    end = next(node["id"] for node in flow["nodes"] if node["type"] == "end")
    flow["nodes"] = [node for node in flow["nodes"] if node["id"] != end]
    for node in flow["nodes"]:
        if node["type"] == "say":
            node["config"]["text"] = ANSWER
    for edge in flow["edges"]:
        if edge["target"] == end:
            edge["target"] = switch
    return cfg


async def login(client: httpx.AsyncClient, credentials: dict) -> None:
    data = check(await client.post("/api/v1/auth/login", json=credentials), 200)
    client.headers["X-CSRF-Token"] = data["csrf_token"]


async def register(client: httpx.AsyncClient, mailbox: httpx.AsyncClient) -> dict:
    credentials = {
        "email": f"load-{uuid4().hex}@example.com",
        "password": secrets.token_urlsafe(24),
    }
    check(await client.post("/api/v1/auth/register", json=credentials), 202)
    async with asyncio.timeout(25):
        while True:
            for message in check(await mailbox.get("/api/v1/messages"), 200)["messages"]:
                if not any(to["Address"] == credentials["email"] for to in message["To"]):
                    continue
                content = check(await mailbox.get(f"/api/v1/message/{message['ID']}"), 200)
                matched = re.search(r"#token=([A-Za-z0-9_-]+)", content["Text"])
                if matched is None:
                    raise RuntimeError("The verification message has no link")
                check(
                    await client.post("/api/v1/auth/verify-email", json={"token": matched[1]}), 204
                )
                await login(client, credentials)
                return credentials
            await asyncio.sleep(0.4)


async def wait_version(client: httpx.AsyncClient, path: str) -> None:
    async with asyncio.timeout(VERSION_TIMEOUT):
        while True:
            version = check(await client.get(path), 200)
            if version["status"] == "ready":
                return
            if version["status"] != "building":
                raise RuntimeError(f"The training job ended with {version['status']}")
            await asyncio.sleep(0.5)


async def train(client: httpx.AsyncClient, agent_path: str, revision: int) -> str:
    version = check(
        await client.post(
            f"{agent_path}/versions",
            headers={"Idempotency-Key": uuid4().hex},
            json={"source_revision": revision},
        ),
        202,
    )
    return version["version_id"]


async def prepare_agent(client: httpx.AsyncClient, org: str, index: int, cfg: dict) -> dict:
    base = f"/api/v1/organizations/{org}"
    agent = check(
        await client.post(
            f"{base}/agents", json={"name": f"Нагрузка {index}", "slug": f"load-{index}"}
        ),
        201,
    )
    path = f"{base}/agents/{agent['id']}"
    lease = check(await client.post(f"{path}/edit-lease", json={"tab_id": str(uuid4())}), 200)
    fence = {
        "X-Edit-Lease": lease["lease_token"],
        "X-Edit-Fence": str(lease["fencing_counter"]),
    }
    saved = check(
        await client.put(
            f"{path}/draft",
            headers={**fence, "If-Match": '"rev-1"'},
            json={"schema_version": 1, "config": cfg},
        ),
        200,
    )
    check(await client.delete(f"{path}/edit-lease", headers=fence), 204)
    revision = saved["revision"]
    version = await train(client, path, revision)
    return {
        "organization": org,
        "id": agent["id"],
        "path": path,
        "revision": revision,
        "version": version,
    }


async def publish(client: httpx.AsyncClient, agent: dict) -> None:
    path, version = agent["path"], agent["version"]
    await wait_version(client, f"{path}/versions/{version}")
    chat = check(await client.post(f"{path}/test-conversations", json={"version_id": version}), 201)
    result = check(
        await client.post(
            f"/api/v1/organizations/{agent['organization']}/test-conversations/{chat['conversation_id']}/messages",
            headers={
                "X-Conversation-Token": chat["conversation_token"],
                "Idempotency-Key": uuid4().hex,
            },
            json={
                "client_message_id": str(uuid4()),
                "revision": chat["revision"],
                "text": PROMPT,
            },
        ),
        200,
    )
    if result["status"] != "waiting_input" or result["messages"] != [ANSWER]:
        raise RuntimeError("The trained agent did not return the expected answer")
    current = await client.get(path)
    check(current, 200)
    check(
        await client.post(
            f"{path}/releases",
            headers={"If-Match": current.headers["ETag"]},
            json={
                "version_id": version,
                "warning_acknowledgement": True,
                "reason": "Проверка нагрузки",
            },
        ),
        201,
    )
    key = check(await client.post(f"{path}/api-keys", json={"name": "Нагрузочный профиль"}), 201)
    agent["key"] = key["secret"]


async def main(args) -> None:
    teams, per_team = (1, 2) if args.smoke else (10, 10)
    cfg = config()
    prefix = uuid4().hex[:10]
    async with (
        httpx.AsyncClient(base_url=args.url, headers={"Origin": args.url}, timeout=40) as client,
        httpx.AsyncClient(base_url=args.mailbox) as mailbox,
    ):
        credentials = await register(client, mailbox)
        agents = []
        for team in range(teams):
            org = check(
                await client.post(
                    "/api/v1/organizations",
                    json={"name": f"Нагрузочный стенд {team + 1}", "slug": f"load-{prefix}-{team}"},
                ),
                201,
            )
            for index in range(per_team):
                agents.append(await prepare_agent(client, org["id"], index + 1, cfg))
        print(f"Queued {len(agents)} training jobs.", flush=True)
        for index, agent in enumerate(agents, start=1):
            await publish(client, agent)
            print(f"Published {index}/{len(agents)} agents.", flush=True)
        payload = {
            "url": args.url,
            "smoke": args.smoke,
            "credentials": credentials,
            "agents": agents,
        }
        descriptor = os.open(args.inventory, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump(payload, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--url", default="http://localhost:18180")
    parser.add_argument("--mailbox", default="http://localhost:18125")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.inventory.exists():
        parser.error("The inventory file already exists")
    arguments.url = arguments.url.rstrip("/")
    asyncio.run(main(arguments))
