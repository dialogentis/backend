from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.config import Settings
from dialogentis.models import Organization
from dialogentis.runtime.core import RuntimeResult
from dialogentis.runtime.history_context import (
    MAX_HISTORY_MESSAGES,
    ContextMessage,
    HistoryContext,
    recent_history,
)
from dialogentis.runtime.history_filters import history_cutoff
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.receipts import ReceiptIdentity, ReceiptPayload, open_receipt, seal_receipt

PURPOSE = b"conversation-history-aes256gcm"


def start_history(db: AsyncSession, row: TestConversation, *, status: str) -> ConversationHistory:
    history = ConversationHistory(
        organization_id=row.organization_id,
        id=row.id,
        agent_id=row.agent_id,
        version_id=row.version_id,
        release_id=row.release_id,
        channel_type="widget" if row.widget_id else "api" if row.api_key_id else "test",
        status=status,
        created_at=row.created_at,
        last_activity_at=row.created_at,
    )
    db.add(history)
    return history


async def record_history(
    db: AsyncSession,
    org: Organization,
    row: TestConversation,
    turn_id: UUID,
    now: datetime,
    result: RuntimeResult,
    settings: Settings,
) -> ReceiptPayload:
    history = await db.get(ConversationHistory, (org.id, row.id))
    if history is None:
        history = start_history(db, row, status=result.state.status)
        await db.flush()
    history.status = result.state.status
    history.last_activity_at = now
    safe = ReceiptPayload.from_result(
        row.revision,
        result.state.status,
        result.history_messages or (),
        result.trace,
        result.error_code,
    )
    intent_id = None
    for event in safe.trace:
        if event.node_type == "switch_intent" and event.event == "transition" and event.output:
            try:
                intent_id = UUID(event.output)
            except ValueError:
                pass
    encrypted = None
    if org.store_message_text and (result.history_input is not None or safe.messages):
        content = safe.model_copy(update={"input_text": result.history_input, "trace": ()})
        encrypted = seal_receipt(
            content, ReceiptIdentity(org.id, row.id, turn_id), settings, purpose=PURPOSE
        )
    db.add(
        HistoryTurn(
            organization_id=org.id,
            conversation_id=row.id,
            sequence=result.state.turn,
            turn_id=turn_id,
            created_at=now,
            intent_id=intent_id,
            error_code=safe.error_code,
            encrypted_messages=encrypted,
            trace=[event.model_dump(mode="json") for event in safe.trace],
        )
    )
    return safe


def read_messages(turn: HistoryTurn, settings: Settings) -> ReceiptPayload | None:
    if turn.encrypted_messages is None:
        return None
    return open_receipt(
        turn.encrypted_messages,
        ReceiptIdentity(turn.organization_id, turn.conversation_id, turn.turn_id),
        settings,
        purpose=PURPOSE,
    )


async def load_dialogue_context(
    db: AsyncSession,
    org: Organization,
    conversation_id: UUID,
    before_sequence: int,
    settings: Settings,
) -> HistoryContext:
    if not org.store_message_text:
        return HistoryContext()
    turns = await db.scalars(
        select(HistoryTurn)
        .where(
            HistoryTurn.organization_id == org.id,
            HistoryTurn.conversation_id == conversation_id,
            HistoryTurn.sequence < before_sequence,
            HistoryTurn.created_at > history_cutoff(org),
            HistoryTurn.encrypted_messages.is_not(None),
        )
        .order_by(HistoryTurn.sequence.desc())
        .limit(MAX_HISTORY_MESSAGES + 1)
    )

    def newest_first():
        for turn in turns:
            saved = read_messages(turn, settings)
            if saved is not None:
                for message in reversed(saved.messages):
                    yield ContextMessage(role="agent", text=message)
                if saved.input_text is not None:
                    yield ContextMessage(role="user", text=saved.input_text)

    return recent_history(newest_first())


async def cleanup_history(
    db: AsyncSession,
    org: Organization,
    *,
    purge_text: bool = False,
    retention_cutoff: datetime | None = None,
) -> None:
    if purge_text:
        await db.execute(
            update(HistoryTurn)
            .where(
                HistoryTurn.organization_id == org.id, HistoryTurn.encrypted_messages.is_not(None)
            )
            .values(encrypted_messages=None)
        )
    cutoff = (
        retention_cutoff
        if retention_cutoff is not None
        else func.now() - timedelta(days=org.retention_days)
    )
    await db.execute(
        delete(HistoryTurn).where(
            HistoryTurn.organization_id == org.id,
            HistoryTurn.created_at <= cutoff,
        )
    )
    await db.execute(
        delete(ConversationHistory).where(
            ConversationHistory.organization_id == org.id,
            ConversationHistory.last_activity_at <= cutoff,
        )
    )


async def apply_history_policy(
    db: AsyncSession,
    org: Organization,
    *,
    retention_cutoff: datetime | None,
    purge_text: bool,
) -> None:
    if retention_cutoff is None:
        return
    receipts = update(RuntimeReceipt).where(
        RuntimeReceipt.organization_id == org.id, RuntimeReceipt.encrypted_response.is_not(None)
    )
    if not purge_text:
        receipts = receipts.where(RuntimeReceipt.created_at <= retention_cutoff)
    await db.execute(receipts.values(encrypted_response=None))
    await cleanup_history(db, org, purge_text=purge_text, retention_cutoff=retention_cutoff)
