from dataclasses import replace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from dialogentis.config import Settings
from dialogentis.runtime.core import TraceEvent
from dialogentis.runtime.receipts import (
    ReceiptError,
    ReceiptIdentity,
    ReceiptPayload,
    open_receipt,
    request_fingerprint,
    seal_receipt,
)


def test_keyed_fingerprint_binds_all_request_parts_and_rejects_bad_text():
    settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
    org, conversation, message = uuid4(), uuid4(), uuid4()
    args = (org, conversation, message, "key", "секретная фраза")
    first = request_fingerprint(settings, *args)
    assert first == request_fingerprint(settings, *args)
    for changed in (
        (uuid4(), conversation, message, "key", args[-1]),
        (org, uuid4(), message, "key", args[-1]),
        (org, conversation, uuid4(), "key", args[-1]),
        (org, conversation, message, "other", args[-1]),
        (org, conversation, message, "key", "другое"),
    ):
        assert request_fingerprint(settings, *changed)[1] != first[1]
    for key, text in ((" ", "text"), ("key", ""), ("key", "\ud800")):
        with pytest.raises(ReceiptError):
            request_fingerprint(settings, org, conversation, message, key, text)


def test_encrypted_response_is_bound_to_tenant_conversation_and_turn():
    settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
    identity = ReceiptIdentity(uuid4(), uuid4(), uuid4())
    payload = ReceiptPayload.from_result(
        2,
        "waiting_input",
        ("SECRET_MESSAGE",),
        (TraceEvent(uuid4(), "say", "transition", "next"),),
        None,
    )
    ciphertext = seal_receipt(payload, identity, settings)
    assert b"SECRET_MESSAGE" not in ciphertext and "SECRET_MESSAGE" not in repr(payload)
    assert ciphertext != seal_receipt(payload, identity, settings)
    assert open_receipt(ciphertext, identity, settings) == payload
    for field in ("organization_id", "conversation_id", "turn_id"):
        with pytest.raises(ReceiptError):
            open_receipt(ciphertext, replace(identity, **{field: uuid4()}), settings)
    with pytest.raises(ReceiptError):
        open_receipt(ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]), identity, settings)
    with pytest.raises(ReceiptError):
        open_receipt(
            ciphertext, identity, settings.model_copy(update={"encryption_key_version": 2})
        )
