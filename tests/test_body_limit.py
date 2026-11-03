import httpx
import pytest
from fastapi import Request

from dialogentis.api import create_app
from dialogentis.config import Settings


async def chunks():
    yield b"123456"
    yield b"789012"


async def small_chunks():
    yield b"abc"
    yield b"def"


@pytest.mark.asyncio
async def test_streaming_body_limit_and_request_id():
    app = create_app(Settings(max_body_bytes=10))

    @app.post("/echo")
    async def echo(request: Request):
        return {"body": (await request.body()).decode()}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        oversized = await client.post("/api/v1/auth/login", content=chunks())
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "payload_too_large"
        assert oversized.json()["error"]["request_id"] == oversized.headers["x-request-id"]
        lied = await client.post(
            "/api/v1/auth/login", headers={"Content-Length": "1"}, content=b"x" * 11
        )
        assert lied.status_code == 413
        within_limit = await client.post("/echo", content=small_chunks())
        assert within_limit.status_code == 200 and within_limit.json() == {"body": "abcdef"}
        ordinary = await client.get("/api/health/live")
        assert ordinary.status_code == 200
