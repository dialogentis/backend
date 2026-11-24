import json
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.telemetry import logger


def test_health_errors_and_request_ids():
    with TestClient(
        create_app(Settings(_env_file=None, database_url=None, broker_url=None))
    ) as client:
        response = client.get("/api/health/live")
        assert response.json() == {"status": "ok"}
        UUID(response.headers["x-request-id"])
        assert client.get("/api/health/ready").status_code == 503
        error = client.get("/api/missing")
        assert error.json()["error"]["code"] == "http_404"
        assert error.json()["error"]["request_id"] == error.headers["x-request-id"]
        assert client.get("/api/openapi.json").status_code == 200


def test_oversized_body_is_rejected():
    with TestClient(create_app(Settings(_env_file=None, max_body_bytes=8))) as client:
        assert client.post("/api/missing", content="123456789").status_code == 413


def test_request_logs_omit_headers_query_and_unexpected_exception_details(monkeypatch):
    records = []
    monkeypatch.setattr(logger, "log", lambda _level, record: records.append(json.loads(record)))
    app = create_app(Settings(_env_file=None, database_url=None, broker_url=None))

    @app.get("/api/failure/{secret}")
    async def failure(secret: str):
        raise RuntimeError(secret)

    with TestClient(app) as client:
        response = client.get(
            "/api/failure/private-path?text=private-query",
            headers={"Authorization": "Bearer private-header"},
        )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
    assert "private-" not in json.dumps(records) and "private-" not in response.text
    assert records[-1]["request_id"] == response.headers["x-request-id"]
    assert records[-1]["operation"] == "failure" and records[-1]["duration_ms"] >= 0
    assert records[-1]["event"] == "request_finished"


def test_production_fails_closed_without_config():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production")
