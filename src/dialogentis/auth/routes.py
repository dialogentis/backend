import asyncio
import json
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from dialogentis.api_errors import error_response
from dialogentis.auth.models import AuthToken, EmailOutbox, RateEvent, Session
from dialogentis.auth.security import (
    COOKIE,
    CurrentUser,
    check_password,
    csrf_token,
    digest,
    dummy_hash,
    hasher,
    rate_limit,
    require_origin,
)
from dialogentis.contracts import InputModel
from dialogentis.dependencies import Database
from dialogentis.models import Membership, User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
NEUTRAL = {"message": "Если адрес доступен, письмо с инструкциями будет отправлено."}


class Credentials(InputModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class EmailInput(InputModel):
    email: EmailStr


class TokenInput(InputModel):
    token: str = Field(min_length=20, max_length=512)


class ResetInput(TokenInput):
    password: str = Field(min_length=12, max_length=128)


class UserView(BaseModel):
    id: UUID
    email: EmailStr
    email_verified: bool


class LoginResult(BaseModel):
    user: UserView
    csrf_token: str


class MembershipView(BaseModel):
    organization_id: UUID
    role: str


class MeResult(LoginResult):
    memberships: list[MembershipView]


def user_json(user: User) -> dict:
    return {"id": str(user.id), "email": user.email, "email_verified": user.email_verified}


async def enqueue_token(db: Database, request: Request, user: User, purpose: str) -> None:
    key = request.app.state.settings.encryption_key
    if key is None:
        raise HTTPException(503, "Почтовый сервис не настроен")
    token = secrets.token_urlsafe(32)
    await db.execute(
        delete(AuthToken).where(AuthToken.user_id == user.id, AuthToken.purpose == purpose)
    )
    db.add(
        AuthToken(
            user_id=user.id,
            purpose=purpose,
            token_hash=digest(token),
            expires_at=datetime.now(UTC) + timedelta(minutes=30 if purpose == "reset" else 1440),
        )
    )
    path = "reset-password" if purpose == "reset" else "verify-email"
    subject = "Восстановление пароля" if purpose == "reset" else "Подтверждение почты"
    url = f"{request.app.state.settings.public_base_url}/{path}#token={token}"
    payload = json.dumps(
        {
            "to": user.email,
            "subject": f"Dialogentis — {subject}",
            "text": (
                f"{subject}\n\nОткройте ссылку:\n{url}\n\n"
                "Если вы не запрашивали это письмо, проигнорируйте его."
            ),
        }
    )
    db.add(
        EmailOutbox(
            encrypted_payload=Fernet(key.get_secret_value().encode())
            .encrypt(payload.encode())
            .decode()
        )
    )


async def consume_token(db: Database, raw: str, purpose: str) -> User:
    token = await db.scalar(
        select(AuthToken)
        .where(
            AuthToken.token_hash == digest(raw),
            AuthToken.purpose == purpose,
            AuthToken.consumed_at.is_(None),
            AuthToken.expires_at > datetime.now(UTC),
        )
        .with_for_update()
    )
    if token is None:
        raise HTTPException(400, "Ссылка недействительна или срок её действия истёк")
    token.consumed_at = datetime.now(UTC)
    user = await db.scalar(select(User).where(User.id == token.user_id).with_for_update())
    if user is None:
        raise HTTPException(400, "Ссылка недействительна")
    return user


@router.post("/register", status_code=202, dependencies=[Depends(require_origin)])
async def register(body: Credentials, request: Request, db: Database):
    email = str(body.email).casefold()
    await rate_limit(db, request, "verification", email, 3, 60)
    await rate_limit(
        db, request, "register-ip", request.client.host if request.client else "unknown", 20, 60
    )
    encoded = await asyncio.to_thread(hasher.hash, body.password)
    result = await db.scalar(
        insert(User)
        .values(email=email, password_hash=encoded, email_verified=False)
        .on_conflict_do_nothing()
        .returning(User.id)
    )
    user = (
        await db.get(User, result)
        if result
        else await db.scalar(select(User).where(User.email == email))
    )
    if user is not None and not user.email_verified:
        await enqueue_token(db, request, user, "verify")
    return NEUTRAL


@router.post("/resend-verification", status_code=202, dependencies=[Depends(require_origin)])
async def resend(body: EmailInput, request: Request, db: Database):
    email = str(body.email).casefold()
    await rate_limit(db, request, "verification", email, 3, 60)
    user = await db.scalar(select(User).where(User.email == email))
    if user is not None and not user.email_verified:
        await enqueue_token(db, request, user, "verify")
    return NEUTRAL


@router.post("/verify-email", status_code=204, dependencies=[Depends(require_origin)])
async def verify(body: TokenInput, db: Database):
    user = await consume_token(db, body.token, "verify")
    user.email_verified = True


@router.post("/login", response_model=LoginResult)
async def login(body: Credentials, request: Request, db: Database, response: Response):
    await require_origin(request)
    email = str(body.email).casefold()
    ip = request.client.host if request.client else "unknown"
    ip_key = await rate_limit(db, request, "login-ip", ip, 20, 15, record=False)
    pair_key = await rate_limit(db, request, "login-pair", f"{ip}:{email}", 5, 15, record=False)
    user = await db.scalar(select(User).where(User.email == email).with_for_update())
    valid = await check_password(user.password_hash if user else dummy_hash, body.password)
    if user is None or not valid:
        db.add_all([RateEvent(subject_hash=ip_key), RateEvent(subject_hash=pair_key)])
        return error_response(request, 401, "invalid_credentials", "Неверная почта или пароль")
    old = request.cookies.get(COOKIE)
    if old:
        await db.execute(delete(Session).where(Session.token_hash == digest(old)))
    token = secrets.token_urlsafe(48)
    now = datetime.now(UTC)
    db.add(
        Session(
            user_id=user.id,
            token_hash=digest(token),
            expires_at=now + timedelta(days=7),
            absolute_expires_at=now + timedelta(days=30),
        )
    )
    response.set_cookie(
        COOKIE,
        token,
        max_age=7 * 86400,
        httponly=True,
        samesite="lax",
        secure=request.app.state.settings.public_base_url.startswith("https://"),
    )
    return {"user": user_json(user), "csrf_token": csrf_token(request, token)}


@router.get("/me", response_model=MeResult)
async def me(request: Request, user: CurrentUser, db: Database):
    memberships = (await db.scalars(select(Membership).where(Membership.user_id == user.id))).all()
    return {
        "user": user_json(user),
        "memberships": [
            {"organization_id": str(m.organization_id), "role": m.role} for m in memberships
        ],
        "csrf_token": csrf_token(request, request.cookies[COOKIE]),
    }


@router.post("/logout", status_code=204)
async def logout(request: Request, user: CurrentUser, db: Database, response: Response):
    await db.execute(delete(Session).where(Session.token_hash == digest(request.cookies[COOKIE])))
    response.delete_cookie(COOKIE, httponly=True, samesite="lax")
    request.state.session_cookie = None


@router.post("/logout-all", status_code=204)
async def logout_all(request: Request, user: CurrentUser, db: Database, response: Response):
    await db.execute(delete(Session).where(Session.user_id == user.id))
    response.delete_cookie(COOKIE, httponly=True, samesite="lax")
    request.state.session_cookie = None


@router.post("/forgot-password", status_code=202, dependencies=[Depends(require_origin)])
async def forgot(body: EmailInput, request: Request, db: Database):
    email = str(body.email).casefold()
    await rate_limit(db, request, "reset", email, 3, 60)
    user = await db.scalar(select(User).where(User.email == email))
    if user is not None:
        await enqueue_token(db, request, user, "reset")
    return NEUTRAL


@router.post("/reset-password", status_code=204, dependencies=[Depends(require_origin)])
async def reset_password(body: ResetInput, request: Request, db: Database, response: Response):
    user = await consume_token(db, body.token, "reset")
    user.password_hash = await asyncio.to_thread(hasher.hash, body.password)
    await db.execute(delete(Session).where(Session.user_id == user.id))
    response.delete_cookie(COOKIE, httponly=True, samesite="lax")
