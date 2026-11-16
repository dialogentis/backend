"""Bounded, tenant-scoped cleanup of expired test-conversation state."""

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.runtime.models import TestConversation


async def cleanup_expired_conversations(db: AsyncSession, organization_id: UUID) -> int:
    ids = (
        await db.scalars(
            select(TestConversation.id)
            .where(
                TestConversation.organization_id == organization_id,
                TestConversation.idle_expires_at <= func.now(),
            )
            .order_by(TestConversation.idle_expires_at, TestConversation.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )
    ).all()
    if not ids:
        return 0
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
    from sqlalchemy import update

    ids = (
        await db.scalars(
            select(TestConversation.id)
            .where(
                TestConversation.organization_id == organization_id,
                TestConversation.processing_token.is_not(None),
                TestConversation.processing_expires_at <= func.now(),
            )
            .order_by(TestConversation.processing_expires_at, TestConversation.id)
            .limit(100)
            .with_for_update(skip_locked=True)
        )
    ).all()
    if not ids:
        return 0
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
    from dialogentis.runtime.receipt_models import RuntimeReceipt

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
