import asyncio
import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from typing import Annotated
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, select, text

from dialogentis.auth.models import RateEvent, Session
from dialogentis.dependencies import Database
from dialogentis.models import User

hasher = PasswordHasher()
dummy_hash = hasher.hash("unusable-account-password-timing-padding")
COOKIE = "dg_session"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def check_password(encoded: str, password: str) -> bool:
    try:
        return await asyncio.to_thread(hasher.verify, encoded, password)
    except VerificationError, InvalidHashError:
        return False


def csrf_token(request: Request, token: str) -> str:
    secret = request.app.state.settings.session_secret
    if secret is None:
        raise HTTPException(503, "Сервис сеансов не настроен")
    return hmac.new(
        secret.get_secret_value().encode(), f"csrf:{token}".encode(), "sha256"
    ).hexdigest()


async def require_origin(request: Request) -> None:
    config = request.app.state.settings
    allowed = {config.public_base_url.rstrip("/")}
    if config.app_env == "development" and urlsplit(config.public_base_url).hostname in {
        "localhost",
        "127.0.0.1",
    }:
        allowed.update(
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (5178, 5180, 18000, 18080)
        )
    if request.headers.get("origin") not in allowed:
        raise HTTPException(403, "Недопустимый источник запроса")


async def current_user(request: Request, db: Database) -> User:
    token = request.cookies.get(COOKIE, "")
    now = datetime.now(UTC)
    session = (
        await db.scalar(
            select(Session)
            .where(
                Session.token_hash == digest(token),
                Session.expires_at > now,
                Session.absolute_expires_at > now,
            )
            .with_for_update()
        )
        if token
        else None
    )
    if session is None:
        raise HTTPException(401, "Войдите в аккаунт")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        await require_origin(request)
        if not hmac.compare_digest(
            request.headers.get("x-csrf-token", ""), csrf_token(request, token)
        ):
            raise HTTPException(403, "Недействительный CSRF-токен")
    user = await db.get(User, session.user_id)
    if user is None:
        raise HTTPException(401, "Войдите в аккаунт")
    await db.execute(text("SELECT set_config('app.user_id', :id, true)"), {"id": str(user.id)})
    session.expires_at = min(now + timedelta(days=7), session.absolute_expires_at)
    request.state.session_cookie = (token, session.expires_at)
    return user


CurrentUser = Annotated[User, Depends(current_user, scope="function")]


async def verified_user(user: CurrentUser) -> User:
    if not user.email_verified:
        raise HTTPException(403, "Подтвердите адрес электронной почты")
    return user


VerifiedUser = Annotated[User, Depends(verified_user, scope="function")]


async def rate_limit(
    db: Database,
    request: Request,
    scope: str,
    value: str,
    limit: int,
    minutes: int,
    *,
    record: bool = True,
) -> str:
    secret = request.app.state.settings.session_secret
    if secret is None:
        raise HTTPException(503, "Сервис сеансов не настроен")
    subject = hmac.new(
        secret.get_secret_value().encode(), f"{scope}:{value}".encode(), "sha256"
    ).hexdigest()
    lock = int.from_bytes(bytes.fromhex(subject[:16]), signed=True)
    await db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
    now = datetime.now(UTC)
    count = await db.scalar(
        select(func.count())
        .select_from(RateEvent)
        .where(
            RateEvent.subject_hash == subject,
            RateEvent.created_at > now - timedelta(minutes=minutes),
        )
    )
    if count is not None and count >= limit:
        raise HTTPException(
            429,
            "Слишком много попыток. Попробуйте позже.",
            headers={"Retry-After": str(minutes * 60)},
        )
    if record:
        db.add(RateEvent(subject_hash=subject))
    return subject
