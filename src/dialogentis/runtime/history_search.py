from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Annotated
from unicodedata import normalize
from uuid import UUID

from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, Field, StringConstraints, ValidationError
from sqlalchemy import and_, or_, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.config import Settings
from dialogentis.contracts import InputModel
from dialogentis.models import Organization
from dialogentis.runtime.core import MASKED
from dialogentis.runtime.history import read_messages
from dialogentis.runtime.history_filters import HistoryFilters, filtered_history, history_cutoff
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.receipts import ReceiptError

SCAN_TURNS = 200
SCAN_BYTES = 8 * 1024 * 1024


class HistorySearchInput(InputModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)] = (
        Field(repr=False)
    )
    filters: HistoryFilters = Field(default_factory=HistoryFilters)
    limit: int = Field(25, ge=1, le=100)
    cursor: str | None = Field(None, min_length=1, max_length=256)


class SearchPosition(BaseModel):
    created_at: AwareDatetime
    conversation_id: UUID
    sequence: int | None = Field(ge=0)

    @classmethod
    def read(cls, cursor: str) -> SearchPosition:
        try:
            return cls.model_validate_json(urlsafe_b64decode(cursor))
        except ValidationError, ValueError:
            raise HTTPException(422, "Некорректная страница поиска") from None

    def encode(self) -> str:
        return urlsafe_b64encode(self.model_dump_json().encode()).decode("ascii")


def normalized(text: str) -> str:
    return normalize("NFKC", text).casefold()


async def search_history(
    db: AsyncSession, org: Organization, settings: Settings, body: HistorySearchInput
) -> tuple[list[ConversationHistory], str | None]:
    if not org.store_message_text:
        raise HTTPException(409, "Сохранение текста отключено")
    wanted = normalized(body.text)
    query = (
        filtered_history(org, body.filters)
        .add_columns(HistoryTurn)
        .join(
            HistoryTurn,
            and_(
                HistoryTurn.organization_id == ConversationHistory.organization_id,
                HistoryTurn.conversation_id == ConversationHistory.id,
            ),
        )
        .where(
            HistoryTurn.created_at > history_cutoff(org),
            HistoryTurn.encrypted_messages.is_not(None),
        )
    )
    if body.cursor is not None:
        cursor = SearchPosition.read(body.cursor)
        current = tuple_(ConversationHistory.created_at, ConversationHistory.id)
        before = (cursor.created_at, cursor.conversation_id)
        query = query.where(
            current < before
            if cursor.sequence is None
            else or_(
                current < before, and_(current == before, HistoryTurn.sequence > cursor.sequence)
            )
        )
    query = query.order_by(
        ConversationHistory.created_at.desc(),
        ConversationHistory.id.desc(),
        HistoryTurn.sequence,
    ).limit(SCAN_TURNS + 1)

    items = []
    matched = set()
    last = None
    total_bytes = 0
    rows = await db.stream(query.execution_options(yield_per=4))
    try:
        for_count = 0
        async for conversation, turn in rows:
            length = len(turn.encrypted_messages or b"")
            if (
                for_count == SCAN_TURNS
                or len(items) == body.limit
                or (for_count and total_bytes + length > SCAN_BYTES)
            ):
                assert last is not None
                return items, last.encode()
            for_count += 1
            total_bytes += length
            if conversation.id not in matched:
                try:
                    payload = read_messages(turn, settings)
                except ReceiptError:
                    raise HTTPException(503, "Не удалось прочитать сообщения") from None
                assert payload is not None
                messages = (payload.input_text or "", *payload.messages)
                if any(wanted in normalized(text.replace(MASKED, "\x00")) for text in messages):
                    items.append(conversation)
                    matched.add(conversation.id)
            last = SearchPosition(
                created_at=conversation.created_at,
                conversation_id=conversation.id,
                sequence=None if conversation.id in matched else turn.sequence,
            )
        return items, None
    finally:
        await rows.close()
