import re
from dataclasses import dataclass
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import any_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.auth.security import digest
from dialogentis.channels.api_key_models import AgentApiKey, KeyScope
from dialogentis.models import Organization


@dataclass(frozen=True)
class ApiAccess:
    id: UUID
    organization_id: UUID
    agent_id: UUID
    scope: KeyScope

    async def authorize(self, db: AsyncSession) -> None:
        # Serialize acceptance with key revocation, publication and tenant deletion.
        await self.lock_organization(db, self.organization_id)
        matched = await db.scalar(
            update(AgentApiKey)
            .where(
                AgentApiKey.id == self.id,
                AgentApiKey.organization_id == self.organization_id,
                AgentApiKey.agent_id == self.agent_id,
                AgentApiKey.revoked_at.is_(None),
                AgentApiKey.expires_at > func.clock_timestamp(),
                any_(AgentApiKey.scopes) == self.scope,
            )
            .values(last_used_at=func.clock_timestamp())
            .returning(AgentApiKey.id)
        )
        if matched is None:
            raise invalid_key()

    async def lock_organization(self, db: AsyncSession, org_id: UUID) -> None:
        if org_id != self.organization_id:
            raise invalid_key()
        org = await db.scalar(
            select(Organization.id)
            .where(Organization.id == org_id, Organization.deletion_requested_at.is_(None))
            .with_for_update()
        )
        if org is None:
            raise invalid_key()


def invalid_key() -> HTTPException:
    return HTTPException(401, "Недействительный ключ API", headers={"WWW-Authenticate": "Bearer"})


async def identify_api_key(db: AsyncSession, credential: str, scope: KeyScope) -> ApiAccess:
    if re.fullmatch(r"dg_[A-Za-z0-9_-]{43}", credential) is None:
        raise invalid_key()
    hashed = digest(credential)
    await db.execute(
        text("SELECT set_config('app.presented_api_key_hash', :hash, true)"), {"hash": hashed}
    )
    row = (
        await db.execute(
            select(AgentApiKey.id, AgentApiKey.organization_id, AgentApiKey.agent_id).where(
                AgentApiKey.token_hash == hashed,
                AgentApiKey.revoked_at.is_(None),
                AgentApiKey.expires_at > func.clock_timestamp(),
                any_(AgentApiKey.scopes) == scope,
            )
        )
    ).one_or_none()
    await db.execute(text("SELECT set_config('app.presented_api_key_hash', '', true)"))
    if row is None:
        raise invalid_key()
    return ApiAccess(row.id, row.organization_id, row.agent_id, scope)
