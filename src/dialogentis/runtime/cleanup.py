"""Bounded, tenant-scoped cleanup of expired test-conversation state."""

from datetime import timedelta
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.models import Organization
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt


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
    expiry_rows = (
        await db.execute(
            select(TestConversation.id, TestConversation.idle_expires_at).where(
                TestConversation.id.in_(ids), TestConversation.organization_id == organization_id
            )
        )
    ).all()
    for conversation_id, expired_at in expiry_rows:
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
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
    for conversation_id, token in (
        await db.execute(
            select(TestConversation.id, TestConversation.processing_token).where(
                TestConversation.id.in_(ids), TestConversation.organization_id == organization_id
            )
        )
    ).all():
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
                RuntimeReceipt.status == "processing",
                RuntimeReceipt.processing_token == token,
            )
            .values(status="unknown")
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
