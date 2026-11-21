import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from dialogentis.audit import cleanup_audit
from dialogentis.db import unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.models import Membership
from dialogentis.organizations import AuditEvent
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_audit_permissions_cursor_details_and_retention(app):
    async with client_for(app) as owner, client_for(app) as admin, client_for(app) as outsider:
        owner_id, org_id, _ = await create_agent(app, owner)
        admin_id, _, _ = await create_agent(app, admin)
        outsider_id, foreign_org, _ = await create_agent(app, outsider)
        base = f"/api/v1/organizations/{org_id}"
        report = f"{base}/audit"
        assert (await outsider.get(report)).status_code == 404
        now = datetime.now(UTC)
        marker, expired, foreign = "unsafe-audit-value", uuid4(), uuid4()
        factory = app.state.session_factory
        async with unit_of_work(factory, organization_id=foreign_org) as db:
            db.add(
                AuditEvent(
                    id=foreign,
                    organization_id=foreign_org,
                    actor_id=outsider_id,
                    action="agent.created",
                    target_id=uuid4(),
                    created_at=now,
                )
            )
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=admin_id, role="admin"))
            db.add(Membership(organization_id=org_id, user_id=outsider_id, role="viewer"))
            db.add(
                AuditEvent(
                    id=expired,
                    organization_id=org_id,
                    actor_id=owner_id,
                    action="agent.archived",
                    target_id=uuid4(),
                    created_at=now - timedelta(days=181),
                )
            )
            for _ in range(2):
                db.add(
                    AuditEvent(
                        organization_id=org_id,
                        actor_id=outsider_id,
                        action="history.export_requested",
                        target_id=uuid4(),
                        created_at=now,
                        details={"format": "json", "secret": marker},
                    )
                )
        assert (await outsider.get(report)).status_code == 403
        assert (await owner.get(report, params={"cursor": str(foreign)})).status_code == 404

        settings = await owner.get(base)
        changed = await admin.patch(
            base,
            json={"history_retention_days": 14, "store_message_text": False, "name": marker},
            headers={"If-Match": settings.headers["etag"]},
        )
        assert changed.status_code == 200
        settings_log = await owner.get(report, params={"action": "organization.settings_changed"})
        assert settings_log.status_code == 200
        (item,) = settings_log.json()["items"]
        assert item["actor_id"] == str(admin_id) and "@" in item["actor_email"]
        assert item["details"] == {"history_retention_days": 14, "store_message_text": False}
        assert marker not in settings_log.text

        cursor_filters = {"action": "history.export_requested", "limit": 1}
        first = (await admin.get(report, params=cursor_filters)).json()
        cursor = first["next_cursor"]
        second = (await admin.get(report, params={**cursor_filters, "cursor": cursor})).json()
        assert second.get("next_cursor") is None
        assert first["items"][0]["id"] != second["items"][0]["id"]
        assert first["items"][0]["details"] == second["items"][0]["details"] == {"format": "json"}
        assert (await owner.delete(f"{base}/members/{outsider_id}")).status_code == 204
        prior = (await owner.get(report, params=cursor_filters)).json()["items"][0]
        assert prior["actor_id"] == str(outsider_id) and prior.get("actor_email") is None
        assert (await owner.get(report, params={"cursor": str(expired)})).status_code == 404

        dispatcher = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
        try:
            async with dispatcher() as db:
                assert (
                    org_id
                    in (await db.scalars(select(AuditEvent.organization_id).distinct())).all()
                )
                with pytest.raises(DBAPIError):
                    await db.execute(select(AuditEvent.details))
        finally:
            await dispatcher.kw["bind"].dispose()
        async with unit_of_work(factory, organization_id=org_id) as db:
            assert await db.get(AuditEvent, expired) is not None
            await cleanup_audit(db, org_id, now)
            assert await db.get(AuditEvent, expired) is None
        async with unit_of_work(factory, organization_id=foreign_org) as db:
            assert await db.get(AuditEvent, foreign) is not None
