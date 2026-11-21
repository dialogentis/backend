import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from dialogentis.artifacts import ArtifactStorage
from dialogentis.db import unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.jobs.tasks import process_job
from dialogentis.prepare_restore import prepare_restore
from dialogentis.runtime.history_models import ConversationHistory
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_exports import mark_published
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_offline_restore_applies_retention_and_recovers_queued_delivery(app):
    factory = app.state.session_factory
    discovery = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    try:
        async with client_for(app) as owner:
            _, org_id, agent_id = await create_agent(app, owner)
            base = f"/api/v1/organizations/{org_id}"
            requested = await owner.post(
                f"{base}/agents/{agent_id}/exports", json={"format": "json"}
            )
            assert requested.status_code == 202
            job_id = UUID(requested.json()["job_id"])
            await mark_published(app, org_id)
            old_event, old_dialogue, kept_dialogue = uuid4(), uuid4(), uuid4()
            now = datetime.now(UTC)
            async with unit_of_work(factory, organization_id=org_id) as db:
                db.add(
                    OutboxEvent(
                        id=old_event,
                        organization_id=org_id,
                        job_id=job_id,
                        available_at=now - timedelta(minutes=5),
                        published_at=now - timedelta(minutes=5),
                    )
                )
                for dialogue, at in (
                    (old_dialogue, now - timedelta(days=31)),
                    (kept_dialogue, now),
                ):
                    db.add(
                        ConversationHistory(
                            organization_id=org_id,
                            id=dialogue,
                            agent_id=agent_id,
                            version_id=uuid4(),
                            channel_type="api",
                            status="completed",
                            created_at=at,
                            last_activity_at=at,
                        )
                    )
            await prepare_restore(
                factory, discovery, ArtifactStorage(app.state.settings.artifact_root)
            )
            async with unit_of_work(factory, organization_id=org_id) as db:
                events = list(
                    await db.scalars(select(OutboxEvent).where(OutboxEvent.job_id == job_id))
                )
                assert {item.id for item in events if item.published_at is not None} == {old_event}
                assert len(events) == 2
                assert await db.get(ConversationHistory, (org_id, old_dialogue)) is None
                assert await db.get(ConversationHistory, (org_id, kept_dialogue)) is not None
            await process_job(factory, job_id, org_id)
            async with unit_of_work(factory, organization_id=org_id) as db:
                job = await db.get(Job, job_id)
                assert job is not None and job.status == "succeeded" and job.attempts == 1
    finally:
        await discovery.kw["bind"].dispose()
