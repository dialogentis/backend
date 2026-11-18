import base64
import hmac
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import any_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.channels.widget_models import WidgetChannel
from dialogentis.config import Settings
from dialogentis.models import Organization

WIDGET_TOKEN_TTL = 30 * 60


def unavailable(status: int = 404) -> HTTPException:
    return HTTPException(status, "Чат временно недоступен")


@dataclass(frozen=True)
class WidgetAccess:
    id: UUID
    organization_id: UUID
    agent_id: UUID
    origin: str
    client_ip: str

    async def lock_organization(self, db: AsyncSession, org_id: UUID) -> None:
        if org_id != self.organization_id:
            raise unavailable()
        active = await db.scalar(
            select(Organization.id)
            .where(Organization.id == org_id, Organization.deletion_requested_at.is_(None))
            .with_for_update()
        )
        if active is None:
            raise unavailable()

    async def authorize(self, db: AsyncSession) -> None:
        await self.lock_organization(db, self.organization_id)
        active = await db.scalar(
            select(WidgetChannel.id).where(
                WidgetChannel.id == self.id,
                WidgetChannel.organization_id == self.organization_id,
                WidgetChannel.agent_id == self.agent_id,
                WidgetChannel.enabled.is_(True),
                any_(WidgetChannel.allowed_origins) == self.origin,
            )
        )
        if active is None:
            raise unavailable()


async def lookup_widget(db: AsyncSession, channel_id: UUID) -> WidgetChannel | None:
    await db.execute(
        text("SELECT set_config('app.presented_widget_id', :id, true)"), {"id": str(channel_id)}
    )
    channel = await db.scalar(select(WidgetChannel).where(WidgetChannel.id == channel_id))
    await db.execute(text("SELECT set_config('app.presented_widget_id', '', true)"))
    return channel


class WidgetGrant(BaseModel):
    channel_id: UUID
    organization_id: UUID
    agent_id: UUID
    conversation_id: UUID
    origin: Annotated[str, Field(max_length=512)]
    credential: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$", repr=False)]

    def access(self, client_ip: str) -> WidgetAccess:
        return WidgetAccess(
            self.channel_id, self.organization_id, self.agent_id, self.origin, client_ip
        )


def _seal(settings: Settings) -> Fernet:
    if settings.encryption_key is None:
        raise unavailable(503)
    try:
        raw = base64.urlsafe_b64decode(settings.encryption_key.get_secret_value())
        if len(raw) != 32:
            raise ValueError("Invalid key size")
    except ValueError:
        raise unavailable(503) from None
    key = hmac.digest(raw, b"dialogentis:widget-session:v1", "sha256")
    return Fernet(base64.urlsafe_b64encode(key))


def issue_widget_token(settings: Settings, claim: WidgetGrant) -> str:
    return "dgw_" + _seal(settings).encrypt(claim.model_dump_json().encode()).decode("ascii")


def read_widget_token(settings: Settings, token: str, conversation_id: UUID) -> WidgetGrant:
    if not token.startswith("dgw_") or len(token) > 2048:
        raise unavailable(401)
    try:
        claim = WidgetGrant.model_validate_json(
            _seal(settings).decrypt(token[4:].encode(), ttl=WIDGET_TOKEN_TTL)
        )
    except InvalidToken, ValidationError:
        raise unavailable(401) from None
    if claim.conversation_id != conversation_id:
        raise unavailable(401)
    return claim
