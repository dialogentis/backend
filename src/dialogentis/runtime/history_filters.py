from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field
from sqlalchemy import select

from dialogentis.contracts import InputModel
from dialogentis.models import Organization
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn

type ChannelType = Literal["test", "api", "widget"]
type HistoryStatus = Literal["waiting_input", "completed", "failed", "expired"]


class HistoryFilters(InputModel):
    agent_id: UUID | None = None
    channel_type: ChannelType | None = None
    status: HistoryStatus | None = None
    start_at: AwareDatetime | None = Field(None, alias="from")
    end_at: AwareDatetime | None = Field(None, alias="to")
    intent_id: UUID | None = None
    error_code: str | None = Field(None, min_length=1, max_length=64)


def history_cutoff(org: Organization) -> datetime:
    return datetime.now(UTC) - timedelta(days=org.retention_days)


def history_query(org: Organization):
    return select(ConversationHistory).where(
        ConversationHistory.organization_id == org.id,
        ConversationHistory.last_activity_at > history_cutoff(org),
    )


def filtered_history(org: Organization, filters: HistoryFilters):
    query = history_query(org)
    for column, value in (
        (ConversationHistory.agent_id, filters.agent_id),
        (ConversationHistory.channel_type, filters.channel_type),
        (ConversationHistory.status, filters.status),
    ):
        if value is not None:
            query = query.where(column == value)
    if filters.start_at is not None:
        query = query.where(ConversationHistory.created_at >= filters.start_at)
    if filters.end_at is not None:
        query = query.where(ConversationHistory.created_at < filters.end_at)
    if filters.intent_id is not None or filters.error_code is not None:
        turn = (
            select(HistoryTurn.turn_id)
            .where(
                HistoryTurn.organization_id == org.id,
                HistoryTurn.conversation_id == ConversationHistory.id,
                HistoryTurn.created_at > history_cutoff(org),
            )
            .correlate(ConversationHistory)
        )
        if filters.intent_id is not None:
            turn = turn.where(HistoryTurn.intent_id == filters.intent_id)
        if filters.error_code is not None:
            turn = turn.where(HistoryTurn.error_code == filters.error_code)
        query = query.where(turn.exists())
    return query
