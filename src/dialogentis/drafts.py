import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field, JsonValue

from dialogentis.agents import agent_and_draft
from dialogentis.auth.security import VerifiedUser, digest
from dialogentis.contracts import InputModel, revision_etag
from dialogentis.dependencies import Database
from dialogentis.models import AgentDraft, User
from dialogentis.organizations import audit, organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["drafts"])
LeaseToken = Annotated[str, Header(alias="X-Edit-Lease", min_length=32, max_length=128)]
LeaseFence = Annotated[int, Header(alias="X-Edit-Fence", ge=1)]
IfMatch = Annotated[str, Header(alias="If-Match")]


class LeaseInput(InputModel):
    tab_id: UUID


class LeaseGrant(BaseModel):
    lease_token: str
    fencing_counter: int
    expires_at: datetime


class LeaseView(BaseModel):
    holder_id: UUID
    holder_name: str
    expires_at: datetime


class Position(InputModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class FlowNode(InputModel):
    id: UUID
    type: Literal[
        "start",
        "switch_intent",
        "say",
        "ask_slot",
        "set_slot",
        "condition",
        "http_action",
        "handoff",
        "return",
        "fallback",
        "end",
    ]
    config: dict[str, JsonValue]
    position: Position


class FlowEdge(InputModel):
    id: UUID
    source: UUID
    source_handle: str = Field(min_length=1, max_length=128)
    target: UUID


class Flow(InputModel):
    schema_version: Literal[1]
    flow_id: UUID
    nodes: list[FlowNode] = Field(max_length=300)
    edges: list[FlowEdge] = Field(max_length=600)
    layout: dict[str, JsonValue]


class DraftConfig(InputModel):
    intents: list[dict[str, JsonValue]] = Field(max_length=100)
    entities: list[dict[str, JsonValue]]
    examples: list[dict[str, JsonValue]] = Field(max_length=13000)
    slots: list[dict[str, JsonValue]]
    flow: Flow


class DraftInput(InputModel):
    schema_version: Literal[1]
    config: DraftConfig


class DraftView(BaseModel):
    revision: int
    schema_version: int
    config: DraftConfig
    validation_summary: None = None
    lease: LeaseView | None


def active(draft: AgentDraft, now: datetime) -> bool:
    return draft.lease_expires_at is not None and draft.lease_expires_at > now


def clear(draft: AgentDraft) -> None:
    draft.lease_holder_id = None
    draft.lease_tab_id = None
    draft.lease_token_hash = None
    draft.lease_expires_at = None


def check_lease(draft: AgentDraft, user: User, token: str, fence: int, now: datetime) -> None:
    if (
        not active(draft, now)
        or draft.lease_holder_id != user.id
        or draft.lease_fence != fence
        or draft.lease_token_hash is None
        or not hmac.compare_digest(draft.lease_token_hash, digest(token))
    ):
        raise HTTPException(423, "Блокировка редактирования утрачена")


async def draft_view(draft: AgentDraft, db: Database) -> DraftView:
    lease = None
    if active(draft, datetime.now(UTC)):
        holder = await db.get(User, draft.lease_holder_id)
        assert holder is not None and draft.lease_expires_at is not None
        local = holder.email.split("@", 1)[0]
        lease = LeaseView(
            holder_id=holder.id, holder_name=f"{local[:1]}***", expires_at=draft.lease_expires_at
        )
    return DraftView(
        revision=draft.revision,
        schema_version=draft.schema_version,
        config=DraftConfig.model_validate(draft.configuration),
        lease=lease,
    )


@router.get("/draft", response_model=DraftView)
async def get_draft(
    org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database, response: Response
):
    await organization_context(db, user, org_id)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    response.headers["ETag"] = revision_etag(draft.revision)
    return await draft_view(draft, db)


@router.post("/edit-lease", response_model=LeaseGrant)
async def acquire(org_id: UUID, agent_id: UUID, body: LeaseInput, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    now = datetime.now(UTC)
    if active(draft, now):
        raise HTTPException(423, "Агент уже редактируется")
    token = secrets.token_urlsafe(32)
    draft.lease_holder_id = user.id
    draft.lease_tab_id = body.tab_id
    draft.lease_token_hash = digest(token)
    draft.lease_fence += 1
    expires_at = now + timedelta(seconds=90)
    draft.lease_expires_at = expires_at
    await db.flush()
    return LeaseGrant(lease_token=token, fencing_counter=draft.lease_fence, expires_at=expires_at)


@router.post("/edit-lease/renew", response_model=LeaseGrant)
async def renew(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    token: LeaseToken,
    fence: LeaseFence,
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    now = datetime.now(UTC)
    check_lease(draft, user, token, fence, now)
    expires_at = now + timedelta(seconds=90)
    draft.lease_expires_at = expires_at
    return LeaseGrant(lease_token=token, fencing_counter=fence, expires_at=expires_at)


@router.delete("/edit-lease", status_code=204)
async def release(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    token: LeaseToken,
    fence: LeaseFence,
):
    await organization_context(db, user, org_id, lock=True)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    check_lease(draft, user, token, fence, datetime.now(UTC))
    clear(draft)


@router.post("/edit-lease/revoke", status_code=204)
async def revoke(org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id, "revoke_lease", lock=True)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    if draft.lease_holder_id is not None:
        clear(draft)
        audit(db, org_id, user, "agent.lease_revoked", agent_id)


@router.put("/draft", response_model=DraftView)
async def save_draft(
    org_id: UUID,
    agent_id: UUID,
    body: DraftInput,
    user: VerifiedUser,
    db: Database,
    response: Response,
    token: LeaseToken,
    fence: LeaseFence,
    if_match: IfMatch,
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    check_lease(draft, user, token, fence, datetime.now(UTC))
    if if_match != revision_etag(draft.revision):
        raise HTTPException(412, "Черновик уже изменён. Обновите страницу")
    config = body.config.model_dump(mode="json")
    try:
        size = len(json.dumps(config, ensure_ascii=False, allow_nan=False).encode())
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, "Некорректные данные черновика") from exc
    if size > 10 * 1024 * 1024:
        raise HTTPException(413, "Черновик превышает 10 МиБ")
    draft.configuration = config
    draft.schema_version = body.schema_version
    draft.revision += 1
    await db.flush()
    response.headers["ETag"] = revision_etag(draft.revision)
    return await draft_view(draft, db)
