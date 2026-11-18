from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, Field
from sqlalchemy import select, tuple_

from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage
from dialogentis.dependencies import Database
from dialogentis.models import Organization
from dialogentis.organizations import organization_context
from dialogentis.runtime.executor import EventView
from dialogentis.runtime.history import read_messages
from dialogentis.runtime.history_filters import (
    ChannelType,
    HistoryFilters,
    HistoryStatus,
    filtered_history,
    history_cutoff,
    history_query,
)
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.history_search import HistorySearchInput, search_history
from dialogentis.runtime.rate_limits import take_slot
from dialogentis.runtime.receipts import ReceiptError

router = APIRouter(prefix="/api/v1/organizations/{org_id}/conversations", tags=["history"])
PageSize = Annotated[int, Query(ge=1, le=100)]
TurnCursor = Annotated[int | None, Query(ge=0)]


class ConversationView(BaseModel):
    id: UUID
    agent_id: UUID
    version_id: UUID
    release_id: UUID | None
    channel_type: ChannelType
    status: HistoryStatus
    created_at: datetime
    last_activity_at: datetime


class ConversationDetail(ConversationView):
    store_message_text: bool


class StoredMessage(BaseModel):
    role: Literal["user", "agent"]
    text: str = Field(repr=False)


class MessageTurn(BaseModel):
    turn_id: UUID
    sequence: int
    created_at: datetime
    messages: list[StoredMessage] = Field(repr=False)


class TraceTurn(BaseModel):
    turn_id: UUID
    sequence: int
    created_at: datetime
    intent_id: UUID | None
    error_code: str | None
    trace: list[EventView]


async def organization(db: Database, user: VerifiedUser, org_id: UUID) -> Organization:
    await organization_context(db, user, org_id)
    organization = await db.get(Organization, org_id)
    assert organization is not None
    return organization


async def conversation(
    db: Database, org: Organization, conversation_id: UUID
) -> ConversationHistory:
    row = await db.scalar(history_query(org).where(ConversationHistory.id == conversation_id))
    if row is None:
        raise HTTPException(404, "Диалог не найден")
    return row


@router.get("", response_model=CursorPage[ConversationView])
async def list_history(
    org_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: PageSize = 25,
    cursor: UUID | None = None,
    agent_id: UUID | None = None,
    channel_type: ChannelType | None = None,
    status: HistoryStatus | None = None,
    start_at: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    end_at: Annotated[AwareDatetime | None, Query(alias="to")] = None,
    intent_id: UUID | None = None,
    error_code: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
):
    org = await organization(db, user, org_id)
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise HTTPException(422, "Дата начала должна предшествовать окончанию периода")
    filters = HistoryFilters.model_validate(
        {
            "agent_id": agent_id,
            "channel_type": channel_type,
            "status": status,
            "from": start_at,
            "to": end_at,
            "intent_id": intent_id,
            "error_code": error_code,
        }
    )
    query = filtered_history(org, filters)
    if cursor is not None:
        before = await conversation(db, org, cursor)
        query = query.where(
            tuple_(ConversationHistory.created_at, ConversationHistory.id)
            < (before.created_at, before.id)
        )
    rows = (
        await db.scalars(
            query.order_by(
                ConversationHistory.created_at.desc(), ConversationHistory.id.desc()
            ).limit(limit + 1)
        )
    ).all()
    return CursorPage[ConversationView](
        items=[ConversationView.model_validate(row, from_attributes=True) for row in rows[:limit]],
        next_cursor=str(rows[limit - 1].id) if len(rows) > limit else None,
    )


@router.post("/search", response_model=CursorPage[ConversationView])
async def find_history(
    org_id: UUID, body: HistorySearchInput, request: Request, user: VerifiedUser, db: Database
):
    org = await organization(db, user, org_id)
    start, end = body.filters.start_at, body.filters.end_at
    if start is not None and end is not None and start >= end:
        raise HTTPException(422, "Дата начала должна предшествовать окончанию периода")
    await take_slot(db, org.id, "history-search:organization", 0.25, 8)
    await take_slot(db, org.id, f"history-search:{user.id}", 0.5, 5)
    rows, next_cursor = await search_history(db, org, request.app.state.settings, body)
    return CursorPage[ConversationView](
        items=[ConversationView.model_validate(row, from_attributes=True) for row in rows],
        next_cursor=next_cursor,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def history_detail(org_id: UUID, conversation_id: UUID, user: VerifiedUser, db: Database):
    org = await organization(db, user, org_id)
    row = await conversation(db, org, conversation_id)
    return ConversationDetail(
        **ConversationView.model_validate(row, from_attributes=True).model_dump(),
        store_message_text=org.store_message_text,
    )


async def turn_page(
    db: Database,
    org: Organization,
    conversation_id: UUID,
    limit: int,
    cursor: int | None,
    *,
    messages: bool,
):
    await conversation(db, org, conversation_id)
    query = select(HistoryTurn).where(
        HistoryTurn.organization_id == org.id,
        HistoryTurn.conversation_id == conversation_id,
        HistoryTurn.created_at > history_cutoff(org),
    )
    if cursor is not None:
        query = query.where(HistoryTurn.sequence > cursor)
    if messages:
        if not org.store_message_text:
            return [], None
        query = query.where(HistoryTurn.encrypted_messages.is_not(None))
    rows = (await db.scalars(query.order_by(HistoryTurn.sequence).limit(limit + 1))).all()
    return rows[:limit], str(rows[limit - 1].sequence) if len(rows) > limit else None


@router.get("/{conversation_id}/messages", response_model=CursorPage[MessageTurn])
async def history_messages(
    org_id: UUID,
    conversation_id: UUID,
    request: Request,
    user: VerifiedUser,
    db: Database,
    limit: PageSize = 25,
    cursor: TurnCursor = None,
):
    org = await organization(db, user, org_id)
    rows, next_cursor = await turn_page(db, org, conversation_id, limit, cursor, messages=True)
    items = []
    for row in rows:
        try:
            data = read_messages(row, request.app.state.settings)
        except ReceiptError:
            raise HTTPException(503, "Не удалось прочитать сообщения") from None
        assert data is not None
        messages = (
            [StoredMessage(role="user", text=data.input_text)]
            if data.input_text is not None
            else []
        )
        messages.extend(StoredMessage(role="agent", text=text) for text in data.messages)
        items.append(
            MessageTurn(
                turn_id=row.turn_id,
                sequence=row.sequence,
                created_at=row.created_at,
                messages=messages,
            )
        )
    return CursorPage[MessageTurn](items=items, next_cursor=next_cursor)


@router.get("/{conversation_id}/trace", response_model=CursorPage[TraceTurn])
async def history_trace(
    org_id: UUID,
    conversation_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: PageSize = 25,
    cursor: TurnCursor = None,
):
    org = await organization(db, user, org_id)
    rows, next_cursor = await turn_page(db, org, conversation_id, limit, cursor, messages=False)
    items = [TraceTurn.model_validate(row, from_attributes=True) for row in rows]
    return CursorPage[TraceTurn](items=items, next_cursor=next_cursor)
