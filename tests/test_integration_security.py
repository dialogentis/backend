from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr, ValidationError

from dialogentis.config import Settings
from dialogentis.integrations.schema import IntegrationInput, safe_base_url, safe_path
from dialogentis.integrations.secrets import IntegrationSecretError, open_secret, seal_secret


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com",
        "https://user:pass@example.com",
        "https://localhost:abc",
        "https://127.0.0.1",
        "https://[::1]",
        "https://169.254.169.254",
        "https://example.com/a/../private",
        "https://example.com/%252e%252e/secret",
        "https://example.com/%2fprivate",
        "https://example.com?x=1",
        "https://example.com\\@evil.com",
    ],
)
def test_reject_invalid_integration_urls(value):
    with pytest.raises(ValueError):
        safe_base_url(value)


@pytest.mark.parametrize(
    "value", ["//evil", "/a/../b", "/%2e%2e/b", "/a%2fb", "/a?x=1", "/a\\b", "/%00"]
)
def test_reject_unsafe_relative_paths(value):
    with pytest.raises(ValueError):
        safe_path(value)


def test_configuration_and_secret_are_bound_to_integration():
    config = IntegrationInput(
        name=" Orders ",
        base_url="https://EXAMPLE.com/api/",
        methods=["GET"],
        paths=["/orders/"],
        secret=SecretStr("secret-value"),
    )
    assert config.name == "Orders" and config.base_url == "https://example.com/api"
    assert "secret-value" not in repr(config)
    with pytest.raises(ValidationError):
        IntegrationInput(
            name="A", base_url="https://example.com", methods=["GET", "GET"], paths=["/x"]
        )
    settings = Settings(_env_file=None, encryption_key=SecretStr(Fernet.generate_key().decode()))
    org, item = uuid4(), uuid4()
    encrypted = seal_secret("secret-value", org, item, settings)
    assert b"secret-value" not in encrypted
    assert open_secret(encrypted, org, item, settings) == "secret-value"
    for wrong_org, wrong_item, value in [
        (uuid4(), item, encrypted),
        (org, uuid4(), encrypted),
        (org, item, encrypted[:-1] + bytes([encrypted[-1] ^ 1])),
    ]:
        with pytest.raises(IntegrationSecretError):
            open_secret(value, wrong_org, wrong_item, settings)
    with pytest.raises(IntegrationSecretError):
        seal_secret("secret\r\ninjected", org, item, settings)
