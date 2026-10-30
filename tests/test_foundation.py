from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dialogentis.api import create_app
from dialogentis.config import Settings
from dialogentis.nlu.smoke import check_nlu


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


def test_production_fails_closed_without_config():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production")


def test_nlu_dependencies_train_and_predict():
    assert check_nlu() == {"intent": "ok", "crf": "ok", "regex": "ok"}
