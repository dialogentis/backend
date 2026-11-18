import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.db import unit_of_work
from dialogentis.jobs.tasks import process_job
from dialogentis.models import Membership
from dialogentis.organizations import AuditEvent
from dialogentis.runtime.history import PURPOSE
from dialogentis.runtime.history_deletion import scrub_history_deletions
from dialogentis.runtime.history_deletion_models import HistoryDeletionTarget
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.receipts import ReceiptIdentity, ReceiptPayload, seal_receipt
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_history_deletion_preserves_later_turns_and_disables_old_exports(app):
    factory = app.state.session_factory
    settings = app.state.settings
    async with client_for(app) as owner, client_for(app) as member:
        _, org_id, agent_id = await create_agent(app, owner)
        member_id, other_org, _ = await create_agent(app, member)
        base = f"/api/v1/organizations/{org_id}"
        history = f"{base}/conversations"
        url = f"{base}/history/deletions"
        exports = f"{base}/history/exports"
        closed, active, kept, new = uuid4(), uuid4(), uuid4(), uuid4()
        before = datetime.now(UTC) - timedelta(minutes=1)

        def dialogue(db, identity, channel, at):
            db.add(
                ConversationHistory(
                    organization_id=org_id,
                    id=identity,
                    agent_id=agent_id,
                    version_id=uuid4(),
                    channel_type=channel,
                    status="completed",
                    created_at=at,
                    last_activity_at=at,
                )
            )

        def turn(db, identity, sequence, at, text, reserved_at=None):
            turn_id = uuid4()
            payload = ReceiptPayload(
                revision=sequence + 1, status="completed", messages=(text,), trace=()
            )
            aad = ReceiptIdentity(org_id, identity, turn_id)
            db.add(
                HistoryTurn(
                    organization_id=org_id,
                    conversation_id=identity,
                    sequence=sequence,
                    turn_id=turn_id,
                    created_at=at,
                    trace=[],
                    encrypted_messages=seal_receipt(payload, aad, settings, purpose=PURPOSE),
                )
            )
            db.add(
                RuntimeReceipt(
                    id=turn_id,
                    organization_id=org_id,
                    conversation_id=identity,
                    credential_hash="a" * 64,
                    client_message_id=uuid4(),
                    idempotency_key_hash=uuid4().hex * 2,
                    request_digest="c" * 64,
                    key_version=1,
                    processing_token=uuid4(),
                    status="completed",
                    revision=sequence + 1,
                    runtime_status="completed",
                    created_at=reserved_at or at,
                    expires_at=at + timedelta(hours=1),
                    encrypted_response=seal_receipt(payload, aad, settings),
                )
            )
            return turn_id

        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=member_id, role="editor"))
            for identity, channel in ((closed, "widget"), (active, "widget"), (kept, "api")):
                dialogue(db, identity, channel, before)
            await db.flush()
            removed_a = turn(db, closed, 0, before, "Удалённая реплика")
            removed_b = turn(db, active, 0, before, "До удаления")
            spared = turn(db, kept, 0, before, "Внешний API")

        export = await owner.post(exports, json={"format": "json"})
        assert export.status_code == 202
        export_id = UUID(export.json()["job_id"])
        await mark_published(app, org_id)
        await process_job(factory, export_id, org_id)
        download = f"{exports}/{export_id}/download"
        assert "Удалённая реплика" in (await owner.get(download)).text

        body = {"confirmation": "delete", "filters": {"channel_type": "widget"}}
        key = {"Idempotency-Key": str(uuid4())}
        assert (await member.post(url, json=body, headers=key)).status_code == 403
        assert (await owner.post(url, json={**body, "confirmation": "skip"})).status_code == 422
        queued = await owner.post(url, json=body, headers=key)
        assert queued.status_code == 202, queued.text
        job_id = UUID(queued.json()["job_id"])
        assert queued.json()["selected_dialogues"] == 2
        assert (await owner.post(url, json=body, headers=key)).json() == queued.json()
        assert (await owner.post(url, json={**body, "filters": {}}, headers=key)).status_code == 409
        assert (
            await member.get(f"/api/v1/organizations/{other_org}/history/deletions/{job_id}")
        ).status_code == 404
        after = datetime.fromisoformat(queued.json()["requested_at"]) + timedelta(milliseconds=1)
        async with unit_of_work(factory, organization_id=org_id) as db:
            row = await db.get(ConversationHistory, (org_id, active))
            assert row is not None
            row.last_activity_at = after
            later = turn(db, active, 1, after, "После подтверждения", reserved_at=before)
            dialogue(db, new, "widget", after)
        await mark_published(app, org_id)
        await process_job(factory, job_id, org_id)
        complete = (await owner.get(f"{url}/{job_id}")).json()
        assert complete["status"] == "succeeded", complete
        assert complete["deleted_dialogues"] == 1
        assert complete["deleted_turns"] == complete["cleared_receipts"] == 2
        assert (await owner.post(url, json=body, headers=key)).json() == complete
        assert (await owner.get(f"{history}/{closed}")).status_code == 404
        for identity in (active, kept, new):
            assert (await owner.get(f"{history}/{identity}")).status_code == 200
        assert (await owner.get(f"{history}/{active}/messages")).json()["items"][0]["messages"] == [
            {"role": "agent", "text": "После подтверждения"},
        ]
        assert (await owner.get(download)).status_code == 404
        async with unit_of_work(factory, organization_id=org_id) as db:
            for identity in (removed_a, removed_b, spared, later):
                row = await db.get(RuntimeReceipt, identity)
                assert row is not None and row.status == "completed"
                assert (row.encrypted_response is None) == (identity in (removed_a, removed_b))
            assert await db.scalar(select(HistoryDeletionTarget)) is None
            events = (
                await db.scalars(select(AuditEvent.action).where(AuditEvent.target_id == job_id))
            ).all()
            assert set(events) == {"history.deletion_requested", "history.deleted"}

        async with unit_of_work(factory, organization_id=org_id) as db:
            membership = await db.get(Membership, (org_id, member_id))
            assert membership is not None
            membership.role = "admin"
        refused = await member.post(url, json={"confirmation": "delete"})
        assert refused.status_code == 202
        refused_id = UUID(refused.json()["job_id"])
        async with unit_of_work(factory, organization_id=org_id) as db:
            membership = await db.get(Membership, (org_id, member_id))
            assert membership is not None
            membership.role = "viewer"
        await mark_published(app, org_id)
        await process_job(factory, refused_id, org_id)
        denied = (await owner.get(f"{url}/{refused_id}")).json()
        assert (
            denied["status"] == "failed"
            and denied["error_code"] == "history_deletion_access_changed"
        )
        assert (await owner.get(f"{history}/{kept}")).status_code == 200
        async with unit_of_work(factory, organization_id=org_id) as db:
            await scrub_history_deletions(db, org_id)
            assert await db.scalar(select(HistoryDeletionTarget)) is None
