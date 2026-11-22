"""Exercise the real API process and an isolated HTTP peer during a forced restart."""

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from sqlalchemy import select

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.integrations.http_client import IntegrationResponse
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.integrations.schema import IntegrationInput
from dialogentis.runtime import live_http
from dialogentis.runtime.cleanup import recover_abandoned_turns
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt


@asynccontextmanager
async def api_process(socket_path, peer_path, owner):
    socket_path.unlink(missing_ok=True)
    with socket_path.with_suffix(".log").open("w+") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.runtime_restart",
            str(socket_path),
            str(peer_path),
            stdout=log,
            stderr=log,
        )
        try:
            async with httpx.AsyncClient(
                base_url="http://test",
                transport=httpx.AsyncHTTPTransport(uds=str(socket_path)),
                headers=owner.headers,
                cookies=owner.cookies,
            ) as client:
                async with asyncio.timeout(10):
                    while True:
                        try:
                            if (await client.get("/api/health/ready")).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if process.returncode is not None:
                            log.seek(0)
                            raise AssertionError(log.read())
                        await asyncio.sleep(0.05)
                yield process, client
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def assert_restart_keeps_post_single(factory, owner, org_id, chat):
    arrived = asyncio.Event()
    disconnected = asyncio.Event()
    requests = []

    async def peer(reader, writer):
        try:
            requests.append((await reader.readuntil(b"\r\n\r\n")).split(b"\r\n")[0])
            arrived.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()

    with TemporaryDirectory(prefix="runtime-restart-") as directory:
        socket_path, peer_path = Path(directory) / "api.sock", Path(directory) / "peer.sock"
        async with await asyncio.start_unix_server(peer, path=peer_path):
            conversation_id = UUID(chat["conversation_id"])
            endpoint = (
                f"/api/v1/organizations/{org_id}/test-conversations/{conversation_id}/messages"
            )
            message_id = uuid4()
            request = {
                "headers": {
                    "X-Conversation-Token": chat["conversation_token"],
                    "Idempotency-Key": "send-before-restart",
                },
                "json": {"client_message_id": str(message_id), "revision": 1, "text": "🛸🛸"},
            }
            async with api_process(socket_path, peer_path, owner) as (first, client):
                send = asyncio.create_task(client.post(endpoint, **request))
                try:
                    await asyncio.wait_for(arrived.wait(), 7)
                    first.kill()
                    assert await first.wait() != 0
                    with pytest.raises(httpx.TransportError):
                        await send
                finally:
                    if not send.done():
                        send.cancel()
                        await asyncio.gather(send, return_exceptions=True)
            await asyncio.wait_for(disconnected.wait(), 7)

            async with api_process(socket_path, peer_path, owner) as (_, client):
                processing = await client.post(endpoint, **request)
                assert processing.status_code == 409 and processing.headers["retry-after"] == "1"
                async with unit_of_work(factory, organization_id=org_id) as db:
                    conversation = await db.get(TestConversation, conversation_id)
                    assert conversation is not None and conversation.processing_expires_at
                    conversation.processing_expires_at = datetime.now(UTC) - timedelta(seconds=1)
                async with unit_of_work(factory, organization_id=org_id) as db:
                    assert await recover_abandoned_turns(db, org_id) >= 1
                    receipt = await db.scalar(
                        select(RuntimeReceipt).where(RuntimeReceipt.client_message_id == message_id)
                    )
                    assert receipt is not None and receipt.status == "unknown"
                    actions = (
                        await db.scalars(
                            select(ActionExecution).where(ActionExecution.receipt_id == receipt.id)
                        )
                    ).all()
                    assert [action.status for action in actions] == ["unknown_result"]
                unknown = await client.post(endpoint, **request)
                assert unknown.status_code == 409 and "неизвестен" in unknown.text
            assert requests == [b"POST /orders/123 HTTP/1.1"]


if __name__ == "__main__":
    api_socket, fake_peer_socket = sys.argv[1:]

    async def send_to_peer(
        config: IntegrationInput,
        method: str,
        path: str,
        query: dict[str, str],
        body: Any,
        *,
        secret: str | None,
        execution_id: UUID,
        supports_idempotency: bool,
        remaining_seconds: float,
    ) -> IntegrationResponse:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=fake_peer_socket), base_url="http://peer"
        ) as client:
            response = await client.request(
                method, path, params=query, json=body, timeout=remaining_seconds
            )
            return IntegrationResponse(response.status_code, response.json())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live_http, "request_json", send_to_peer)
        uvicorn.run(
            create_app(Settings(public_base_url="http://test")), uds=api_socket, access_log=False
        )
