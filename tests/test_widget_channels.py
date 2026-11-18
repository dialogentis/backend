import os
from uuid import UUID

import pytest
from sqlalchemy import select

from dialogentis.channels.widget_models import WidgetChannel
from dialogentis.channels.widget_origins import widget_origin
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.models import Membership
from dialogentis.organizations import AuditEvent
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent


def test_widget_origins_use_exact_authorities_and_development_loopback():
    assert widget_origin(" https://КОМАНДА.рф:443/ ", development=False) == (
        "https://xn--80aalwkhe.xn--p1ai"
    )
    assert widget_origin("https://faß.example", development=False) == "https://xn--fa-hia.example"
    assert (
        widget_origin("https://example.org:8443", development=False) == "https://example.org:8443"
    )
    assert widget_origin("http://[0:0:0:0:0:0:0:1]:5178", development=True) == "http://[::1]:5178"
    assert (
        widget_origin("http://shop.localhost:5178/", development=True)
        == "http://shop.localhost:5178"
    )
    for value in (
        "*",
        "null",
        "https://*.example.org",
        "https://example.org;",
        "https://example.org	",
        "https://example.org/path",
        "https://example.org/?",
        "https://example.org/#",
        "https://user@example.org",
        "https://example.org:0",
        "https://example.org:65536",
        "https://example.org\\example.net",
        "https://example.org；frame-src",
        "https://[2001:db8::1]example.org",
        "http://example.org",
        "https://localhost",
        "https://shop.localhost.",
        "https://127.5.6.7",
        "https://[::ffff:127.0.0.1]",
        "https://127.1",
        "https://0x7f000001",
    ):
        with pytest.raises(ValueError):
            widget_origin(value, development=False)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose")
async def test_widget_settings_permissions_concurrency_and_deletion(app):
    factory = app.state.session_factory
    async with client_for(app) as owner, client_for(app) as member, client_for(app) as anonymous:
        _, org_id, agent_id = await create_agent(app, owner)
        user_id, other_org, other_agent = await create_agent(app, member)
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=user_id, role="editor"))
        base = f"/api/v1/organizations/{org_id}/agents/{agent_id}"
        path = f"{base}/widget-channels"
        body = {
            "name": " Поддержка сайта ",
            "allowed_origins": ["https://SHOP.example:443", "https://shop.example/"],
            "welcome_text": "Добрый день",
            "accent_color": "#AB3C57",
        }
        assert (await anonymous.get(path)).status_code == 401
        assert (await member.get(path)).status_code == 403
        assert (await member.post(path, json=body)).status_code == 403
        assert (await owner.post(path, json={**body, "allowed_origins": ["*"]})).status_code == 422
        assert (await owner.post(path, json={**body, "accent_color": "red"})).status_code == 422
        created = await owner.post(path, json=body)
        assert created.status_code == 201, created.text
        widget = created.json()
        widget_id = UUID(widget["id"])
        item = f"{path}/{widget_id}"
        assert widget["allowed_origins"] == ["https://shop.example"]
        assert widget["name"] == "Поддержка сайта" and widget["accent_color"] == "#ab3c57"
        assert widget["position"] == "bottom-right" and widget["enabled"]
        assert (await owner.get(path)).json()["items"] == [widget]
        assert (await owner.get(item)).headers["etag"] == '"rev-1"'
        wrong = (
            f"/api/v1/organizations/{other_org}/agents/{other_agent}/widget-channels/{widget_id}"
        )
        assert (await member.get(wrong)).status_code == 404
        async with unit_of_work(factory, organization_id=other_org) as db:
            assert await db.get(WidgetChannel, widget_id) is None
        async with unit_of_work(factory) as db:
            assert await db.get(WidgetChannel, widget_id) is None
        async with unit_of_work(factory, organization_id=org_id) as db:
            membership = await db.get(Membership, (org_id, user_id))
            assert membership is not None
            membership.role = "admin"
        changes = {"position": "bottom-left", "allowed_origins": ["https://shop.example:8443"]}
        assert (await member.patch(item, json=changes)).status_code == 428
        updated = await member.patch(item, json=changes, headers={"If-Match": '"rev-1"'})
        assert updated.status_code == 200, updated.text
        assert updated.json()["allowed_origins"] == ["https://shop.example:8443"]
        assert updated.json()["position"] == "bottom-left"
        assert (
            await member.patch(item, json=changes, headers={"If-Match": '"rev-1"'})
        ).status_code == 412
        for _ in range(2):
            assert (await member.post(f"{item}/disable")).status_code == 204
        record = await owner.get(item)
        assert record.json()["enabled"] is False and record.headers["etag"] == '"rev-3"'
        assert (await owner.post(f"{base}/archive")).status_code == 200
        assert (
            await member.patch(item, json={"enabled": True}, headers={"If-Match": '"rev-3"'})
        ).status_code == 409
        async with unit_of_work(factory, organization_id=org_id) as db:
            events = (
                await db.scalars(select(AuditEvent.action).where(AuditEvent.target_id == widget_id))
            ).all()
            assert sorted(events) == ["widget.created", "widget.disabled", "widget.updated"]
            await purge_tenant(db, org_id)
        async with unit_of_work(factory, organization_id=org_id) as db:
            assert await db.get(WidgetChannel, widget_id) is None
