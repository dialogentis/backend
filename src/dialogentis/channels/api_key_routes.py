from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import AwareDatetime, BaseModel, Field, StringConstraints
from sqlalchemy import select, tuple_

from dialogentis.agents import agent_and_draft
from dialogentis.auth.security import VerifiedUser, digest
from dialogentis.channels.api_key_models import AgentApiKey, KeyScope
from dialogentis.contracts import CursorPage, InputModel
from dialogentis.dependencies import Database
from dialogentis.organizations import audit, organization_context

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/agents/{agent_id}/api-keys", tags=["api-keys"]
)


class ApiKeyInput(InputModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    expires_at: AwareDatetime | None = None


class ApiKeyView(BaseModel):
    id: UUID
    agent_id: UUID
    name: str
    prefix: str
    scopes: list[KeyScope]
    created_by: UUID
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class CreatedApiKey(ApiKeyView):
    secret: str = Field(repr=False)


def key_query(org_id: UUID, agent_id: UUID):
    return select(AgentApiKey).where(
        AgentApiKey.organization_id == org_id, AgentApiKey.agent_id == agent_id
    )


@router.get("", response_model=CursorPage[ApiKeyView])
async def list_api_keys(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: UUID | None = None,
):
    await organization_context(db, user, org_id, "manage_channels")
    await agent_and_draft(db, org_id, agent_id)
    query = key_query(org_id, agent_id)
    if cursor is not None:
        before = await db.scalar(query.where(AgentApiKey.id == cursor))
        if before is None:
            raise HTTPException(404, "Ключ не найден")
        query = query.where(
            tuple_(AgentApiKey.created_at, AgentApiKey.id) < tuple_(before.created_at, before.id)
        )
    rows = (
        await db.scalars(
            query.order_by(AgentApiKey.created_at.desc(), AgentApiKey.id.desc()).limit(limit + 1)
        )
    ).all()
    return CursorPage[ApiKeyView](
        items=[ApiKeyView.model_validate(row, from_attributes=True) for row in rows[:limit]],
        next_cursor=str(rows[limit - 1].id) if len(rows) > limit else None,
    )


@router.post("", response_model=CreatedApiKey, status_code=201)
async def create_api_key(
    org_id: UUID, agent_id: UUID, body: ApiKeyInput, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_channels", lock=True)
    agent, _ = await agent_and_draft(db, org_id, agent_id)
    if agent.status != "active":
        raise HTTPException(409, "Сначала восстановите агента из архива")
    now = datetime.now(UTC)
    expiry = body.expires_at or (now + timedelta(days=90))
    if not now < expiry <= now + timedelta(days=365):
        raise HTTPException(422, "Срок действия ключа должен истекать в следующие 365 дней")
    secret = "dg_" + token_urlsafe(32)
    key = AgentApiKey(
        organization_id=org_id,
        agent_id=agent_id,
        name=body.name,
        prefix=secret[:12],
        token_hash=digest(secret),
        scopes=["conversation:create", "message:send"],
        created_by=user.id,
        created_at=now,
        expires_at=expiry,
    )
    db.add(key)
    await db.flush()
    audit(db, org_id, user, "api_key.created", key.id, details={"agent_id": str(agent_id)})
    return CreatedApiKey(
        **ApiKeyView.model_validate(key, from_attributes=True).model_dump(), secret=secret
    )


@router.post("/{key_id}/revoke", status_code=204)
async def revoke_api_key(
    org_id: UUID, agent_id: UUID, key_id: UUID, user: VerifiedUser, db: Database
) -> None:
    await organization_context(db, user, org_id, "manage_channels", lock=True)
    key = await db.scalar(key_query(org_id, agent_id).where(AgentApiKey.id == key_id))
    if key is None:
        raise HTTPException(404, "Ключ не найден")
    if key.revoked_at is None:
        key.revoked_at = datetime.now(UTC)
        audit(db, org_id, user, "api_key.revoked", key.id, details={"agent_id": str(agent_id)})
