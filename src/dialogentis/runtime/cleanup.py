"""Bounded, tenant-scoped cleanup of expired test-conversation state."""

from datetime import timedelta
from uuid import UUID

from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.analytics import record_completed
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.models import Organization
from dialogentis.runtime.history_models import ConversationHistory
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt


async def _lock_due_organization(
    db: AsyncSession, organization_id: UUID, due: ColumnElement[bool]
) -> bool:
    return (
        await db.scalar(
            select(Organization.id)
            .where(
                Organization.id == organization_id,
                Organization.deletion_requested_at.is_(None),
                select(TestConversation.id)
                .where(TestConversation.organization_id == organization_id, due)
                .exists(),
            )
            .with_for_update()
        )
        is not None
    )


async def cleanup_expired_conversations(db: AsyncSession, organization_id: UUID) -> int:
    due = TestConversation.idle_expires_at <= func.now()
    if not await _lock_due_organization(db, organization_id, due):
        return 0
    rows = (
        await db.scalars(
            select(TestConversation)
            .where(
                TestConversation.organization_id == organization_id,
                due,
            )
            .order_by(TestConversation.idle_expires_at, TestConversation.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )
    ).all()
    if not rows:
        return 0
    by_id = {row.id: row for row in rows}
    ids = list(by_id)
    expired = await db.execute(
        update(RuntimeReceipt)
        .where(
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.conversation_id.in_(ids),
            RuntimeReceipt.status == "processing",
        )
        .values(status="unknown")
        .returning(RuntimeReceipt.id, RuntimeReceipt.conversation_id, RuntimeReceipt.created_at)
    )
    for turn_id, conversation_id, accepted_at in expired.all():
        row = by_id[conversation_id]
        await record_completed(db, row, turn_id, accepted_at, row.idle_expires_at, None)
    await db.execute(
        update(ActionExecution)
        .where(
            ActionExecution.organization_id == organization_id,
            ActionExecution.conversation_id.in_(ids),
            ActionExecution.status == "started",
        )
        .values(status="unknown_result", finished_at=func.now())
    )
    await db.execute(
        update(ConversationHistory)
        .where(
            ConversationHistory.organization_id == organization_id,
            ConversationHistory.id.in_(ids),
            ConversationHistory.status == "waiting_input",
        )
        .values(status="expired")
    )
    for row in rows:
        expired_at = row.idle_expires_at
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == row.id,
                RuntimeReceipt.expires_at > expired_at + timedelta(hours=24),
            )
            .values(expires_at=expired_at + timedelta(hours=24))
        )
    deleted = await db.scalars(
        delete(TestConversation)
        .where(
            TestConversation.organization_id == organization_id,
            TestConversation.id.in_(ids),
            TestConversation.idle_expires_at <= func.now(),
        )
        .returning(TestConversation.id)
    )
    return len(deleted.all())


async def recover_abandoned_turns(db: AsyncSession, organization_id: UUID) -> int:
    """Fail expired reservations without replaying a possibly executed turn."""
    due = TestConversation.processing_expires_at <= func.now()
    if not await _lock_due_organization(db, organization_id, due):
        return 0
    rows = (
        await db.scalars(
            select(TestConversation)
            .where(
                TestConversation.organization_id == organization_id,
                TestConversation.processing_token.is_not(None),
                due,
            )
            .order_by(TestConversation.processing_expires_at, TestConversation.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )
    ).all()
    if not rows:
        return 0
    ids = [row.id for row in rows]
    await db.execute(
        update(ConversationHistory)
        .where(
            ConversationHistory.organization_id == organization_id,
            ConversationHistory.id.in_(ids),
            ConversationHistory.status == "waiting_input",
        )
        .values(status="failed")
    )
    for row in rows:
        conversation_id, token = row.id, row.processing_token
        ended_at = row.processing_expires_at
        assert ended_at is not None
        expired = await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
                RuntimeReceipt.status == "processing",
                RuntimeReceipt.processing_token == token,
            )
            .values(status="unknown")
            .returning(RuntimeReceipt.id, RuntimeReceipt.created_at)
        )
        for turn_id, accepted_at in expired.all():
            await record_completed(db, row, turn_id, accepted_at, ended_at, None)
        await db.execute(
            update(ActionExecution)
            .where(
                ActionExecution.organization_id == organization_id,
                ActionExecution.conversation_id == conversation_id,
                ActionExecution.processing_token == token,
                ActionExecution.status == "started",
            )
            .values(status="unknown_result", finished_at=func.now())
        )
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
                RuntimeReceipt.expires_at > func.now() + timedelta(hours=24),
            )
            .values(expires_at=func.now() + timedelta(hours=24))
        )
    updated = await db.scalars(
        update(TestConversation)
        .where(
            TestConversation.organization_id == organization_id,
            TestConversation.id.in_(ids),
            TestConversation.processing_expires_at <= func.now(),
        )
        .values(
            checkpoint=None,
            processing_token=None,
            processing_expires_at=None,
            revision=TestConversation.revision + 1,
        )
        .returning(TestConversation.id)
    )
    return len(updated.all())


async def cleanup_expired_receipts(db: AsyncSession, organization_id: UUID) -> int:
    organization = await db.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    if organization is None:
        return 0
    await db.execute(
        update(RuntimeReceipt)
        .where(
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.encrypted_response.is_not(None),
            RuntimeReceipt.created_at <= func.now() - timedelta(days=organization.retention_days),
        )
        .values(encrypted_response=None)
    )
    ids = (
        await db.scalars(
            select(RuntimeReceipt.id)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.expires_at <= func.now(),
            )
            .order_by(RuntimeReceipt.expires_at, RuntimeReceipt.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )
    ).all()
    if not ids:
        return 0
    deleted = await db.scalars(
        delete(RuntimeReceipt)
        .where(
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.id.in_(ids),
            RuntimeReceipt.expires_at <= func.now(),
        )
        .returning(RuntimeReceipt.id)
    )
    return len(deleted.all())
