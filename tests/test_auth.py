import asyncio
import os
import re
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from dialogentis.api import create_app
from dialogentis.auth.mailer import dispatch_once
from dialogentis.auth.models import AuthToken
from dialogentis.auth.security import COOKIE, digest
from dialogentis.config import Settings
from dialogentis.db import unit_of_work

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Run with Compose test profile",
    ),
]
PASSWORD = "тестовый пароль 12345"


@pytest.fixture
async def app():
    app = create_app(Settings(public_base_url="http://test"))
    async with app.router.lifespan_context(app):
        yield app


def client_for(app):
    address = "fd00:" + ":".join(uuid4().hex[i : i + 4] for i in range(0, 28, 4))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(address, 50000)),
        base_url="http://test",
        headers={"Origin": "http://test"},
    )


async def mail_token(app, email: str, subject: str) -> str:
    await dispatch_once(app.state.settings, app.state.session_factory)
    async with httpx.AsyncClient(base_url="http://mailpit:8025") as mail:
        for _ in range(50):
            result = (await mail.get("/api/v1/messages", params={"limit": 200})).json()
            for message in result["messages"]:
                if (
                    any(recipient["Address"] == email for recipient in message["To"])
                    and subject in message["Subject"]
                ):
                    body = (await mail.get(f"/api/v1/message/{message['ID']}")).json()["Text"]
                    match = re.search(r"token=([A-Za-z0-9_-]+)", body)
                    assert match is not None
                    return match[1]
            await asyncio.sleep(0.1)
    raise AssertionError("Email was not delivered to Mailpit")


async def account(app, client, *, verify=True):
    email = f"account-{uuid4()}@example.com"
    result = await client.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert result.status_code == 202, result.text
    if verify:
        token = await mail_token(app, email, "Подтверждение")
        assert (
            await client.post("/api/v1/auth/verify-email", json={"token": token})
        ).status_code == 204
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return email, response.json()["user"]["id"]


async def test_verification_reset_and_revocation(app):
    async with client_for(app) as client:
        email, _ = await account(app, client, verify=False)
        assert not (await client.get("/api/v1/auth/me")).json()["user"]["email_verified"]
        assert (await client.get("/api/v1/organizations")).status_code == 403
        token = await mail_token(app, email, "Подтверждение")
        async with unit_of_work(app.state.session_factory) as db:
            stored = await db.scalar(select(AuthToken).where(AuthToken.token_hash == digest(token)))
            assert stored is not None and stored.token_hash != token
        assert (
            await client.post("/api/v1/auth/verify-email", json={"token": token})
        ).status_code == 204
        assert (
            await client.post("/api/v1/auth/verify-email", json={"token": token})
        ).status_code == 400
        assert (await client.get("/api/v1/organizations")).status_code == 200
        assert (
            await client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "wrong"})
        ).status_code == 403
        assert (
            await client.post("/api/v1/auth/logout", headers={"Origin": "https://evil.example"})
        ).status_code == 403
        cookie = client.cookies.get(COOKIE)
        response = await client.post("/api/v1/auth/forgot-password", json={"email": email})
        unknown = await client.post(
            "/api/v1/auth/forgot-password", json={"email": f"missing-{uuid4()}@example.com"}
        )
        assert response.status_code == unknown.status_code == 202
        assert response.json() == unknown.json()
        token = await mail_token(app, email, "Восстановление")
        body = {"token": token, "password": "новый пароль 56789"}
        assert (await client.post("/api/v1/auth/reset-password", json=body)).status_code == 204
        assert (await client.post("/api/v1/auth/reset-password", json=body)).status_code == 400
        async with client_for(app) as old_client:
            old_client.cookies.set(COOKIE, cookie)
            assert (await old_client.get("/api/v1/auth/me")).status_code == 401
        response = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": body["password"]}
        )
        assert response.status_code == 200
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=lax" in response.headers["set-cookie"]
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        assert (await client.post("/api/v1/auth/logout-all")).status_code == 204
        assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_rate_limit_persists_failed_attempts(app):
    async with client_for(app) as client:
        body = {"email": f"missing-{uuid4()}@example.com", "password": PASSWORD}
        for _ in range(5):
            assert (await client.post("/api/v1/auth/login", json=body)).status_code == 401
        result = await client.post("/api/v1/auth/login", json=body)
        assert result.status_code == 429
        assert int(result.headers["retry-after"]) > 0
        result = await client.post("/api/v1/auth/register", json={**body, "password": "pa55!"})
        assert result.status_code == 422
        assert "pa55!" not in result.text
        assert '"input"' not in result.text


async def test_invitations_roles_and_transfer(app):
    async with client_for(app) as owner, client_for(app) as guest:
        _, owner_id = await account(app, owner)
        email, guest_id = await account(app, guest)
        result = await owner.post(
            "/api/v1/organizations", json={"name": "Первая", "slug": f"org-{uuid4()}"}
        )
        assert result.status_code == 201, result.text
        org = result.json()["id"]
        assert (await owner.get("/api/v1/organizations")).json()["items"][0]["id"] == org
        assert (await guest.get(f"/api/v1/organizations/{org}")).status_code == 404
        assert (await owner.post(f"/api/v1/organizations/{org}/leave")).status_code == 409
        result = await owner.post(
            f"/api/v1/organizations/{org}/invitations", json={"email": email, "role": "editor"}
        )
        assert result.status_code == 201, result.text
        token = await mail_token(app, email, "Приглашение")
        assert (
            await owner.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 400
        result = await guest.post("/api/v1/invitations/accept", json={"token": token})
        assert result.status_code == 200, result.text
        assert (
            await guest.post("/api/v1/invitations/accept", json={"token": token})
        ).status_code == 400
        for role, allowed in [("viewer", False), ("editor", False), ("admin", True)]:
            result = await owner.patch(
                f"/api/v1/organizations/{org}/members/{guest_id}", json={"role": role}
            )
            assert result.status_code == 200
            assert (await guest.get(f"/api/v1/organizations/{org}")).status_code == 200
            assert (await guest.get(f"/api/v1/organizations/{org}/members")).status_code == (
                200 if allowed else 403
            )
        assert (
            await guest.patch(
                f"/api/v1/organizations/{org}/members/{owner_id}", json={"role": "viewer"}
            )
        ).status_code == 403
        admin_email = f"admin-{uuid4()}@example.com"
        admin_invite = await owner.post(
            f"/api/v1/organizations/{org}/invitations",
            json={"email": admin_email, "role": "admin"},
        )
        assert admin_invite.status_code == 201
        invitation_id = admin_invite.json()["id"]
        path = f"/api/v1/organizations/{org}/invitations/{invitation_id}"
        assert (await guest.delete(path)).status_code == 403
        assert (await guest.post(f"{path}/resend")).status_code == 403
        assert (
            await guest.post(
                f"/api/v1/organizations/{org}/invitations",
                json={"email": admin_email, "role": "editor"},
            )
        ).status_code == 403
        assert (await owner.delete(path)).status_code == 204
        result = await owner.post(
            f"/api/v1/organizations/{org}/transfer-ownership",
            json={"new_owner_user_id": guest_id, "password": PASSWORD},
        )
        assert result.status_code == 204, result.text
        assert (await guest.get(f"/api/v1/organizations/{org}")).json()["role"] == "owner"
        assert (await owner.post(f"/api/v1/organizations/{org}/leave")).status_code == 204
        assert (await owner.get(f"/api/v1/organizations/{org}")).status_code == 404


async def test_invitation_resend_revoke_and_token_expiry(app):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import text

    from dialogentis.organizations import Invitation

    async with client_for(app) as owner, client_for(app) as recipient:
        _, owner_id = await account(app, owner)
        email, _ = await account(app, recipient)
        result = await owner.post(
            "/api/v1/organizations",
            json={
                "name": "Вторая",
                "slug": f"org-{uuid4()}",
            },
        )
        assert result.status_code == 201
        org_id = result.json()["id"]
        result = await owner.post(
            f"/api/v1/organizations/{org_id}/invitations", json={"email": email, "role": "viewer"}
        )
        assert result.status_code == 201
        invitation_id = result.json()["id"]
        old_token = await mail_token(app, email, "Приглашение")
        assert (
            await owner.post(f"/api/v1/organizations/{org_id}/invitations/{invitation_id}/resend")
        ).status_code == 202
        new_token = await mail_token(app, email, "Приглашение")
        assert (
            await recipient.post("/api/v1/invitations/accept", json={"token": old_token})
        ).status_code == 400
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            invitation = await db.get(Invitation, invitation_id)
            assert invitation is not None
            assert invitation.token_hash != digest(old_token)
            invitation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert (
            await recipient.post("/api/v1/invitations/accept", json={"token": new_token})
        ).status_code == 400
        assert (
            await owner.delete(f"/api/v1/organizations/{org_id}/invitations/{invitation_id}")
        ).status_code == 204
        assert (
            await owner.post(f"/api/v1/organizations/{org_id}/invitations/{invitation_id}/resend")
        ).status_code == 409
        async with unit_of_work(app.state.session_factory) as db:
            await db.execute(
                text(
                    "UPDATE sessions SET absolute_expires_at = now() - interval '1 second' "
                    "WHERE user_id = :id"
                ),
                {"id": owner_id},
            )
        assert (await owner.get("/api/v1/auth/me")).status_code == 401
