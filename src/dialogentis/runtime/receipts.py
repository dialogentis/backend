"""Cryptographic primitives for durable idempotency receipts.

Only authenticated, tenant-scoped callers may use these primitives. They do not
reserve a turn or persist data; those operations must share a database transaction.
"""

import base64
import hashlib
import hmac
import os
import struct
from dataclasses import dataclass
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dialogentis.config import Settings
from dialogentis.runtime.core import TraceEvent

MAGIC = b"DGR1"
MAX_RECEIPT = 2 * 1024 * 1024


class ReceiptError(ValueError):
    def __init__(self, code: str = "invalid_receipt"):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ReceiptIdentity:
    organization_id: UUID
    conversation_id: UUID
    turn_id: UUID

    def aad(self, header: bytes) -> bytes:
        return header + self.organization_id.bytes + self.conversation_id.bytes + self.turn_id.bytes


def _key(settings: Settings, purpose: bytes) -> bytes:
    if settings.encryption_key is None or not 1 <= settings.encryption_key_version <= 2**32 - 1:
        raise ReceiptError("receipt_key_unavailable")
    try:
        encoded = settings.encryption_key.get_secret_value().encode("ascii")
        Fernet(encoded)
        secret = base64.urlsafe_b64decode(encoded)
    except ValueError, TypeError, UnicodeError:
        raise ReceiptError("receipt_key_unavailable") from None
    return hmac.digest(secret, b"dialogentis:runtime:receipt:" + purpose + b":v1", "sha256")


def start_identity(
    settings: Settings, organization_id: UUID, user_id: UUID, agent_id: UUID, key: str
) -> tuple[UUID, UUID, str]:
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= 128
        or any(not 33 <= ord(char) <= 126 for char in key)
    ):
        raise ReceiptError("invalid_idempotency_key")
    payload = organization_id.bytes + user_id.bytes + agent_id.bytes + key.encode("ascii")
    conversation = UUID(
        bytes=hashlib.sha256(b"dialogentis:start:conversation:v1" + payload).digest()[:16]
    )
    message = UUID(bytes=hashlib.sha256(b"dialogentis:start:message:v1" + payload).digest()[:16])
    token = (
        base64.urlsafe_b64encode(hmac.digest(_key(settings, b"start-token"), payload, "sha256"))
        .rstrip(b"=")
        .decode("ascii")
    )
    return conversation, message, token


def request_fingerprint(
    settings: Settings,
    organization_id: UUID,
    conversation_id: UUID,
    client_message_id: UUID,
    idempotency_key: str,
    text: str,
) -> tuple[str, str]:
    if (
        not isinstance(idempotency_key, str)
        or not 1 <= len(idempotency_key) <= 128
        or any(not 33 <= ord(c) <= 126 for c in idempotency_key)
    ):
        raise ReceiptError("invalid_idempotency_key")
    if not isinstance(text, str) or not 1 <= len(text) <= 2000:
        raise ReceiptError("invalid_message")
    try:
        message = text.encode("utf-8")
    except UnicodeError:
        raise ReceiptError("invalid_message") from None
    key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    payload = (
        organization_id.bytes
        + conversation_id.bytes
        + client_message_id.bytes
        + bytes.fromhex(key_hash)
        + message
    )
    return key_hash, hmac.digest(_key(settings, b"request-hmac"), payload, "sha256").hex()


class _Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: UUID
    node_type: str = Field(max_length=32)
    event: str = Field(max_length=64)
    output: str | None = Field(default=None, max_length=128)
    fallback_reasons: tuple[str, ...] = Field(default=(), max_length=10)
    version_id: UUID | None = None


class ReceiptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = Field(ge=1)
    status: str = Field(pattern=r"^(waiting_input|completed|failed)$")
    messages: tuple[str, ...] = Field(max_length=52, repr=False)
    trace: tuple[_Trace, ...] = Field(max_length=120, repr=False)
    error_code: str | None = Field(default=None, max_length=64)
    input_text: str | None = Field(default=None, max_length=4000, repr=False)

    @classmethod
    def from_result(
        cls,
        revision: int,
        status: str,
        messages: tuple[str, ...],
        trace: tuple[TraceEvent, ...],
        error_code: str | None,
    ) -> ReceiptPayload:
        return cls.model_validate(
            {
                "revision": revision,
                "status": status,
                "messages": messages,
                "trace": [
                    {
                        "node_id": item.node_id,
                        "node_type": item.node_type,
                        "event": item.event,
                        "output": item.output,
                        "fallback_reasons": item.fallback_reasons,
                        "version_id": item.version_id,
                    }
                    for item in trace
                ],
                "error_code": error_code,
            }
        )


def seal_receipt(
    payload: ReceiptPayload,
    identity: ReceiptIdentity,
    settings: Settings,
    *,
    purpose: bytes = b"payload-aes256gcm",
) -> bytes:
    key = _key(settings, purpose)
    try:
        plain = payload.model_dump_json().encode("utf-8")
    except ValueError, UnicodeError:
        raise ReceiptError() from None
    if len(plain) > MAX_RECEIPT:
        raise ReceiptError("receipt_too_large")
    header = MAGIC + struct.pack(">I", settings.encryption_key_version)
    nonce = os.urandom(12)
    return header + nonce + AESGCM(key).encrypt(nonce, plain, identity.aad(header))


def open_receipt(
    content: bytes,
    identity: ReceiptIdentity,
    settings: Settings,
    *,
    purpose: bytes = b"payload-aes256gcm",
) -> ReceiptPayload:
    key = _key(settings, purpose)
    if not isinstance(content, bytes) or not 36 <= len(content) <= MAX_RECEIPT + 36:
        raise ReceiptError()
    header = content[:8]
    if header[:4] != MAGIC or struct.unpack(">I", header[4:])[0] != settings.encryption_key_version:
        raise ReceiptError()
    try:
        plain = AESGCM(key).decrypt(content[8:20], content[20:], identity.aad(header))
        return ReceiptPayload.model_validate_json(plain)
    except InvalidTag, ValidationError, ValueError:
        raise ReceiptError() from None
