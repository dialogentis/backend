import csv
import io
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import cleanup_expired, read_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.db import unit_of_work
from dialogentis.jobs.tasks import process_job
from dialogentis.models import Membership
from dialogentis.organizations import AuditEvent
from dialogentis.runtime.history import PURPOSE
from dialogentis.runtime.history_export_models import HistoryExportTask
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.receipts import ReceiptIdentity, ReceiptPayload, seal_receipt
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_history_exports_permissions_formats_revocation_and_privacy(app):
    factory = app.state.session_factory
    settings = app.state.settings
    async with client_for(app) as owner, client_for(app) as member:
        _, org_id, agent_id = await create_agent(app, owner)
        member_id, other_org, _ = await create_agent(app, member)
        base = f"/api/v1/organizations/{org_id}"
        url = f"{base}/history/exports"
        filters = {"channel_type": "widget", "agent_id": str(agent_id)}
        now = datetime.now(UTC) - timedelta(minutes=1)
        formula = "\ufeff ＝СУММ(1;2)"
        dialogue, turn = uuid4(), uuid4()
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=member_id, role="viewer"))
            for identity, channel in ((dialogue, "widget"), (uuid4(), "test")):
                db.add(
                    ConversationHistory(
                        organization_id=org_id,
                        id=identity,
                        agent_id=agent_id,
                        version_id=uuid4(),
                        channel_type=channel,
                        status="completed",
                        created_at=now,
                        last_activity_at=now,
                    )
                )
            await db.flush()
            encrypted = seal_receipt(
                ReceiptPayload(
                    revision=2,
                    status="completed",
                    messages=("Доставка [скрыто]",),
                    trace=(),
                    input_text=formula,
                ),
                ReceiptIdentity(org_id, dialogue, turn),
                settings,
                purpose=PURPOSE,
            )
            db.add(
                HistoryTurn(
                    organization_id=org_id,
                    conversation_id=dialogue,
                    sequence=1,
                    turn_id=turn,
                    created_at=now,
                    trace=[],
                    encrypted_messages=encrypted,
                )
            )

        async def start(client, format):
            response = await client.post(url, json={"format": format, "filters": filters})
            assert response.status_code == 202, response.text
            assert response.json()["status"] == "queued"
            await mark_published(app, org_id)
            return UUID(response.json()["job_id"])

        async def role(value):
            async with unit_of_work(factory, organization_id=org_id) as db:
                membership = await db.get(Membership, (org_id, member_id))
                assert membership is not None
                membership.role = value

        assert (await member.post(url, json={"format": "json"})).status_code == 403
        first = await start(owner, "json")
        download = f"{url}/{first}/download"
        assert (await owner.get(download)).status_code == 404
        await process_job(factory, first, org_id)
        job = (await owner.get(f"{url}/{first}")).json()
        assert job["status"] == "succeeded", job
        until = datetime.fromisoformat(job["download_until"])
        assert timedelta(minutes=14) < until - datetime.now(UTC) <= timedelta(minutes=15)
        assert (await member.get(download)).status_code == 403
        wrong = f"/api/v1/organizations/{other_org}/history/exports/{first}/download"
        assert (await member.get(wrong)).status_code == 404
        received = await owner.get(download)
        assert received.status_code == 200
        assert received.headers["cache-control"] == "no-store"
        assert "dialogentis-history.json" in received.headers["content-disposition"]
        export = received.json()
        assert export["filters"] == filters and export["generated_at"]
        assert [record["conversation_id"] for record in export["records"]] == [str(dialogue)]
        assert export["records"][0]["user_text"] == formula
        assert export["records"][0]["agent_messages"] == ["Доставка [скрыто]"]
        storage = ArtifactStorage(settings.artifact_root)
        async with unit_of_work(factory, organization_id=org_id) as db:
            task = await db.get(HistoryExportTask, first)
            assert task is not None and task.artifact_id is not None
            stored = await read_artifact(db, storage, org_id, task.artifact_id, max_bytes=50_000)
            assert stored.startswith(b"DGE1") and formula.encode() not in stored
            artifact = await db.get(Artifact, task.artifact_id)
            assert artifact is not None and artifact.expires_at is not None
            assert artifact.expires_at - until == timedelta(minutes=45)
            events = (
                await db.scalars(select(AuditEvent.action).where(AuditEvent.target_id == first))
            ).all()
            assert set(events) == {"history.export_requested", "history.export_downloaded"}

        await role("admin")
        second = await start(member, "csv")
        await process_job(factory, second, org_id)
        csv_url = f"{url}/{second}/download"
        response = await member.get(csv_url)
        assert response.status_code == 200, response.text
        metadata, row = list(csv.DictReader(io.StringIO(response.text)))
        assert metadata["record_type"] == "export" and metadata["generated_at"]
        assert row["record_type"] == "turn" and row["user_text"] == "'" + formula
        assert row["agent_messages"] == '["Доставка [скрыто]"]'
        await role("viewer")
        assert (await member.get(csv_url)).status_code == 403
        await role("admin")
        pending = await start(member, "csv")
        await role("viewer")
        await process_job(factory, pending, org_id)
        failed = (await owner.get(f"{url}/{pending}")).json()
        assert failed["status"] == "failed"
        assert failed["error_code"] == "history_export_access_changed"

        async with unit_of_work(factory, organization_id=org_id) as db:
            task = await db.get(HistoryExportTask, second)
            assert task is not None and task.artifact_id is not None
            task.download_until = now
            artifact = await db.get(Artifact, task.artifact_id)
            assert artifact is not None
            artifact.expires_at = now
        assert (await owner.get(csv_url)).status_code == 404
        async with unit_of_work(factory, organization_id=org_id) as db:
            assert await cleanup_expired(db, storage, org_id) == 1
            task = await db.get(HistoryExportTask, second)
            assert task is not None and task.artifact_id is None
        etag = (await owner.get(base)).headers["etag"]
        updated = await owner.patch(
            base,
            json={"store_message_text": False, "purge_existing_text": False},
            headers={"If-Match": etag},
        )
        assert updated.status_code == 200
        assert (await owner.get(download)).status_code == 404
        without_text = await start(owner, "json")
        await process_job(factory, without_text, org_id)
        response = await owner.get(f"{url}/{without_text}/download")
        assert response.status_code == 200
        assert response.json()["store_message_text"] is False
        assert formula not in response.text and "[скрыто]" not in response.text
