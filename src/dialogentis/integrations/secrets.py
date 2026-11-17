"""Encrypt integration credentials bound to their tenant and integration."""

import base64
import hmac
import os
import struct
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from dialogentis.config import Settings

MAGIC = b"DGI1"


class IntegrationSecretError(ValueError):
    def __init__(self):
        super().__init__("invalid_integration_secret")


def _key(settings: Settings) -> bytes:
    if settings.encryption_key is None or not 1 <= settings.encryption_key_version <= 2**32 - 1:
        raise IntegrationSecretError()
    try:
        encoded = settings.encryption_key.get_secret_value().encode("ascii")
        Fernet(encoded)
        secret = base64.urlsafe_b64decode(encoded)
    except ValueError, TypeError, UnicodeError:
        raise IntegrationSecretError() from None
    return hmac.digest(secret, b"dialogentis:integration:secret:aes256gcm:v1", "sha256")


def seal_secret(
    value: str, organization_id: UUID, integration_id: UUID, settings: Settings
) -> bytes:
    try:
        plain = value.encode("utf-8")
    except UnicodeError:
        raise IntegrationSecretError() from None
    if not 1 <= len(plain) <= 2048 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise IntegrationSecretError()
    key = _key(settings)
    header = MAGIC + struct.pack(">I", settings.encryption_key_version)
    nonce = os.urandom(12)
    return (
        header
        + nonce
        + AESGCM(key).encrypt(nonce, plain, header + organization_id.bytes + integration_id.bytes)
    )


def open_secret(
    data: bytes, organization_id: UUID, integration_id: UUID, settings: Settings
) -> str:
    if not isinstance(data, bytes) or not 37 <= len(data) <= 2084:
        raise IntegrationSecretError()
    header = data[:8]
    if header[:4] != MAGIC or struct.unpack(">I", header[4:])[0] != settings.encryption_key_version:
        raise IntegrationSecretError()
    try:
        return (
            AESGCM(_key(settings))
            .decrypt(data[8:20], data[20:], header + organization_id.bytes + integration_id.bytes)
            .decode("utf-8")
        )
    except InvalidTag, ValueError, UnicodeError:
        raise IntegrationSecretError() from None
