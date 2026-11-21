from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, StrictBool, StringConstraints, ValidationError
from sqlalchemy import and_, delete, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage
from dialogentis.dependencies import Database
from dialogentis.models import Membership, User
from dialogentis.organizations import AuditEvent, Role, organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/audit", tags=["audit"])
RETENTION = timedelta(days=180)
type Code = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]

FIELDS = {
    "organization.settings_changed": {
        "history_retention_days",
        "store_message_text",
        "purge_existing_text",
    },
    "member.role_changed": {"role_before", "role_after"},
    "agent.published": {"agent_id", "previous_release_id", "reason", "acknowledged_warnings"},
    "agent.rolled_back": {"agent_id", "previous_release_id", "reason"},
    "api_key.created": {"agent_id"},
    "api_key.revoked": {"agent_id"},
    "widget.created": {"agent_id"},
    "widget.updated": {"fields"},
    "history.export_requested": {"format"},
    "history.export_downloaded": {"format"},
    "history.deletion_requested": {"selected_dialogues"},
    "history.deleted": {"dialogues", "turns", "cleared_receipts"},
}


class AuditDetails(BaseModel):
    history_retention_days: int | None = Field(None, ge=1, le=30)
    store_message_text: StrictBool | None = None
    purge_existing_text: StrictBool | None = None
    role_before: Role | None = None
    role_after: Role | None = None
    agent_id: UUID | None = None
    previous_release_id: UUID | None = None
    reason: str | None = Field(None, max_length=500)
    acknowledged_warnings: list[Code] | None = Field(None, max_length=100)
    fields: list[Code] | None = Field(None, max_length=32)
    format: Literal["json", "csv"] | None = None
    selected_dialogues: int | None = Field(None, ge=0)
    dialogues: int | None = Field(None, ge=0)
    turns: int | None = Field(None, ge=0)
    cleared_receipts: int | None = Field(None, ge=0)


class AuditView(BaseModel):
    id: UUID
    action: str
    created_at: datetime
    actor_id: UUID
    actor_email: str | None = None
    target_id: UUID
    details: AuditDetails


def _details(event: AuditEvent) -> AuditDetails:
    safe = {
        key: value for key, value in event.details.items() if key in FIELDS.get(event.action, ())
    }
    try:
        return AuditDetails.model_validate(safe)
    except ValidationError:
        return AuditDetails()


@router.get("", response_model=CursorPage[AuditView], response_model_exclude_none=True)
async def list_audit(
    org_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    cursor: UUID | None = None,
    action: Annotated[str | None, Query(pattern=r"^[a-z][a-z0-9_.]{1,63}$")] = None,
):
    await organization_context(db, user, org_id, "view_audit")
    visible = [
        AuditEvent.organization_id == org_id,
        AuditEvent.created_at > datetime.now(UTC) - RETENTION,
    ]
    if action is not None:
        visible.append(AuditEvent.action == action)
    query = (
        select(AuditEvent, User.email)
        .outerjoin(
            Membership,
            and_(Membership.organization_id == org_id, Membership.user_id == AuditEvent.actor_id),
        )
        .outerjoin(User, User.id == Membership.user_id)
        .where(*visible)
    )
    if cursor is not None:
        boundary = await db.scalar(select(AuditEvent).where(*visible, AuditEvent.id == cursor))
        if boundary is None:
            raise HTTPException(404, "Страница журнала недоступна")
        query = query.where(
            tuple_(AuditEvent.created_at, AuditEvent.id) < (boundary.created_at, cursor)
        )
    rows = (
        await db.execute(
            query.order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(limit + 1)
        )
    ).all()
    return CursorPage[AuditView](
        items=[
            AuditView(
                id=event.id,
                action=event.action,
                created_at=event.created_at,
                actor_id=event.actor_id,
                actor_email=email,
                target_id=event.target_id,
                details=_details(event),
            )
            for event, email in rows[:limit]
        ],
        next_cursor=str(rows[limit - 1][0].id) if len(rows) > limit else None,
    )


async def cleanup_audit(db: AsyncSession, org_id: UUID, now: datetime) -> None:
    await db.execute(
        delete(AuditEvent).where(
            AuditEvent.organization_id == org_id, AuditEvent.created_at <= now - RETENTION
        )
    )
