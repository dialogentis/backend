"""Authenticated encryption for internal runtime checkpoints.

Callers must separately enforce tenant membership, expiration and replay protection.
"""

import base64
import hmac
import os
import struct
from dataclasses import dataclass
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import ValidationError

from dialogentis.config import Settings
from dialogentis.runtime.core import RuntimeState

MAGIC = b"DGC1"
NONCE_BYTES = 12
MAX_PLAINTEXT_BYTES = 2 * 1024 * 1024
HEADER_BYTES = len(MAGIC) + 4


class CheckpointError(ValueError):
    """A safe error that never contains serialized state or user input."""

    def __init__(self):
        super().__init__("invalid_checkpoint")


@dataclass(frozen=True)
class CheckpointIdentity:
    organization_id: UUID
    agent_id: UUID
    version_id: UUID
    conversation_id: UUID

    def associated_data(self, header: bytes) -> bytes:
        return header + b"".join(
            item.bytes
            for item in (self.organization_id, self.agent_id, self.version_id, self.conversation_id)
        )


def _key(settings: Settings) -> bytes:
    if settings.encryption_key is None or not 1 <= settings.encryption_key_version <= 2**32 - 1:
        raise CheckpointError()
    try:
        encoded = settings.encryption_key.get_secret_value().encode("ascii")
        Fernet(encoded)
        secret = base64.urlsafe_b64decode(encoded)
    except ValueError, TypeError, UnicodeError:
        raise CheckpointError() from None
    return hmac.digest(secret, b"dialogentis:runtime:checkpoint:aes256gcm:v1", "sha256")


def encrypt_checkpoint(
    state: RuntimeState, identity: CheckpointIdentity, settings: Settings
) -> bytes:
    key = _key(settings)
    if state.version_id != identity.version_id:
        raise CheckpointError()
    try:
        plain = state.model_dump_json().encode("utf-8")
    except ValueError, UnicodeError:
        raise CheckpointError() from None
    if len(plain) > MAX_PLAINTEXT_BYTES:
        raise CheckpointError()
    header = MAGIC + struct.pack(">I", settings.encryption_key_version)
    nonce = os.urandom(NONCE_BYTES)
    return header + nonce + AESGCM(key).encrypt(nonce, plain, identity.associated_data(header))


def decrypt_checkpoint(
    data: bytes, identity: CheckpointIdentity, settings: Settings
) -> RuntimeState:
    key = _key(settings)
    if (
        not isinstance(data, bytes)
        or len(data) < HEADER_BYTES + NONCE_BYTES + 16
        or len(data) > HEADER_BYTES + NONCE_BYTES + MAX_PLAINTEXT_BYTES + 16
    ):
        raise CheckpointError()
    header = data[:HEADER_BYTES]
    if (
        header[: len(MAGIC)] != MAGIC
        or struct.unpack(">I", header[len(MAGIC) :])[0] != settings.encryption_key_version
    ):
        raise CheckpointError()
    nonce = data[HEADER_BYTES : HEADER_BYTES + NONCE_BYTES]
    try:
        plain = AESGCM(key).decrypt(
            nonce, data[HEADER_BYTES + NONCE_BYTES :], identity.associated_data(header)
        )
        state = RuntimeState.model_validate_json(plain)
    except InvalidTag, ValidationError, ValueError:
        raise CheckpointError() from None
    if state.version_id != identity.version_id:
        raise CheckpointError()
    return state
