import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from dialogentis.api import create_app
from dialogentis.config import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Run with Compose test profile",
    ),
]


async def test_database_and_api_readiness(tmp_path, monkeypatch):
    config = Settings()
    assert config.database_url is not None
    async with await psycopg.AsyncConnection.connect(
        config.database_url.get_secret_value()
    ) as connection:
        row = await (await connection.execute("SELECT current_user")).fetchone()
        assert row == ("dialogentis_app",)
    with TestClient(create_app(config)) as client:
        response = client.get("/api/health/ready")
        assert response.status_code == 200, response.json()
        assert response.json()["checks"] == {
            "database": "ok",
            "schema_revision": "ok",
            "artifacts": "ok",
        }
    with TestClient(create_app(config.model_copy(update={"broker_url": None}))) as client:
        assert client.get("/api/health/ready").status_code == 200
    with TestClient(
        create_app(config.model_copy(update={"artifact_root": tmp_path / "missing"}))
    ) as client:
        result = client.get("/api/health/ready")
        assert result.status_code == 503
        assert result.json()["checks"] == {
            "database": "ok",
            "schema_revision": "ok",
            "artifacts": "unavailable",
        }
    monkeypatch.setattr("dialogentis.health.expected_schema", lambda: "incompatible")
    with TestClient(create_app(config)) as client:
        result = client.get("/api/health/ready")
        assert result.status_code == 503
        assert result.json()["checks"] == {
            "database": "ok",
            "schema_revision": "unavailable",
            "artifacts": "ok",
        }


async def test_taskiq_rabbitmq_manual_ack(tmp_path: Path):
    from tests.broker_tasks import broker, probe

    output = tmp_path / "worker.log"
    with output.open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "taskiq",
            "worker",
            "tests.broker_tasks:broker",
            "--workers",
            "1",
            "--ack-type",
            "manual",
            "--max-async-tasks",
            "1",
            env={**os.environ, "TEST_OUTPUT_DIR": str(tmp_path)},
            stdout=log,
            stderr=log,
        )
        try:
            await broker.startup()
            assert broker.write_channel is not None
            assert broker.write_channel.publisher_confirms
            nonce = uuid4()
            await probe.kiq(nonce)
            for _ in range(100):
                if (tmp_path / str(nonce)).exists():
                    break
                assert process.returncode is None, output.read_text()
                await asyncio.sleep(0.1)
            assert (tmp_path / str(nonce)).read_text() == "processed", output.read_text()
        finally:
            await broker.shutdown()
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=10)
