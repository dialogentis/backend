import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from dialogentis.config import Settings


@pytest.fixture
def production():
    return {
        "app_env": "production",
        "public_base_url": "https://dialogentis.example.org",
        "database_url": "postgresql://app@postgres/dialogentis",
        "broker_url": "amqp://app@rabbitmq/",
        "session_secret": "a" * 64,
        "encryption_key": Fernet.generate_key().decode(),
        "smtp_host": "smtp.example.org",
        "smtp_port": 587,
        "smtp_from": "Dialogentis <no-reply@example.org>",
        "artifact_root": "/app/artifacts",
        "deletion_journal_root": "/app/deletion-journal",
    }


def test_production_accepts_starttls_relay_and_authenticated_smtp(production):
    assert Settings(_env_file=None, **production).smtp_port == 587
    assert (
        Settings(
            _env_file=None, **production, smtp_user="mailer", smtp_password="test-password"
        ).smtp_user
        == "mailer"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("public_base_url", "http://dialogentis.example.org"),
        ("public_base_url", "https://dialogentis.example.org/path"),
        ("public_base_url", "https://dialogentis.example.org:443"),
        ("public_base_url", "https://user@dialogentis.example.org"),
        ("public_base_url", "https://dialogentis.example.org/"),
        ("smtp_host", "mailpit"),
        ("smtp_from", "invalid"),
        ("smtp_from", "no-reply@localhost"),
        ("smtp_user", "mailer"),
        ("smtp_port", 65536),
    ],
)
def test_production_rejects_ambiguous_origin_and_incomplete_smtp(production, field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**production, field: value})
