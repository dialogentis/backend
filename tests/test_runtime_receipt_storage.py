import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError, IntegrityError

from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.maintenance import reconcile_once
from dialogentis.jobs.models import OutboxEvent
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.receipts import ReceiptIdentity, ReceiptPayload, open_receipt, seal_receipt
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_receipt_isolation_uniqueness_retention_and_cleanup(app):
    async with client_for(app) as owner, client_for(app) as stranger:
        user_id, org_id, _ = await create_agent(app, owner)
        _, other_org, _ = await create_agent(app, stranger)
        conversation_id, turn_id, message_id, token = uuid4(), uuid4(), uuid4(), uuid4()
        now = datetime.now(UTC)
        settings = Settings(encryption_key=SecretStr(Fernet.generate_key().decode()))
        identity = ReceiptIdentity(org_id, conversation_id, turn_id)
        payload = ReceiptPayload.from_result(2, "waiting_input", ("PRIVATE_RESPONSE",), (), None)
        ciphertext = seal_receipt(payload, identity, settings)

        def make(**changes):
            return RuntimeReceipt(
                **(
                    {
                        "id": uuid4(),
                        "organization_id": org_id,
                        "conversation_id": conversation_id,
                        "created_by": user_id,
                        "credential_hash": "a" * 64,
                        "client_message_id": uuid4(),
                        "idempotency_key_hash": "b" * 64,
                        "request_digest": "c" * 64,
                        "key_version": 1,
                        "processing_token": token,
                        "status": "completed",
                        "revision": 2,
                        "runtime_status": "waiting_input",
                        "encrypted_response": ciphertext,
                        "expires_at": now + timedelta(hours=24),
                    }
                    | changes
                )
            )

        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(make(id=turn_id, client_message_id=message_id))
            await db.execute(delete(OutboxEvent).where(OutboxEvent.organization_id == org_id))
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(RuntimeReceipt, turn_id)
            assert row is not None and row.encrypted_response is not None
            assert b"PRIVATE_RESPONSE" not in row.encrypted_response
            assert open_receipt(row.encrypted_response, identity, settings) == payload
        for bad in (
            {"client_message_id": message_id, "idempotency_key_hash": "d" * 64},
            {"idempotency_key_hash": "b" * 64},
            {"idempotency_key_hash": "e" * 64, "revision": None},
            {"idempotency_key_hash": "e" * 64, "encrypted_response": b"plaintext"},
        ):
            with pytest.raises(IntegrityError):
                async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                    db.add(make(**bad))
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(RuntimeReceipt, turn_id) is None
        async with unit_of_work(app.state.session_factory) as db:
            assert await db.get(RuntimeReceipt, turn_id) is None
        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            async with dispatcher() as db:
                assert (
                    org_id
                    in (await db.scalars(select(RuntimeReceipt.organization_id).distinct())).all()
                )
            with pytest.raises(DBAPIError):
                async with dispatcher() as db:
                    await db.execute(select(RuntimeReceipt.encrypted_response))
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                row = await db.get(RuntimeReceipt, turn_id)
                assert row is not None
                row.expires_at = now - timedelta(seconds=1)
            assert await reconcile_once(
                app.state.session_factory, dispatcher, datetime(2042, 1, 1, tzinfo=UTC), frozenset()
            )
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                assert await db.get(RuntimeReceipt, turn_id) is None
                db.add(make(idempotency_key_hash="e" * 64))
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                await purge_tenant(db, org_id)
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                assert not (await db.scalars(select(RuntimeReceipt))).all()
        finally:
            await dispatcher.kw["bind"].dispose()
