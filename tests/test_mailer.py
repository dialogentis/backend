import os
from datetime import UTC, datetime, timedelta

import pytest

from dialogentis.auth.mailer import MAX_DELIVERY_ATTEMPTS, dispatch_once
from dialogentis.auth.models import EmailOutbox
from dialogentis.db import unit_of_work
from dialogentis.legacy_mail import has_untagged_invitation_mail
from tests.test_auth import app as app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_exhausted_mail_drops_content_and_expires(app):
    factory = app.state.session_factory
    config = app.state.settings
    now = datetime.now(UTC)
    async with unit_of_work(factory) as db:
        last_attempt = EmailOutbox(
            encrypted_payload="unreadable-mail",
            attempts=MAX_DELIVERY_ATTEMPTS - 1,
            available_at=now - timedelta(days=1),
        )
        legacy = EmailOutbox(
            encrypted_payload="old-undelivered-mail",
            attempts=MAX_DELIVERY_ATTEMPTS,
            available_at=now - timedelta(minutes=5),
        )
        db.add_all([last_attempt, legacy])
        await db.flush()
        ids = (last_attempt.id, legacy.id)

    await dispatch_once(config, factory)
    async with unit_of_work(factory) as db:
        for row_id in ids:
            row = await db.get(EmailOutbox, row_id)
            assert row is not None and row.encrypted_payload == ""
            assert row.attempts == MAX_DELIVERY_ATTEMPTS and row.sent_at is None
            row.available_at = now - timedelta(days=2)
        assert not await has_untagged_invitation_mail(db, config.encryption_key.get_secret_value())

    await dispatch_once(config, factory)
    async with unit_of_work(factory) as db:
        for row_id in ids:
            assert await db.get(EmailOutbox, row_id) is None
