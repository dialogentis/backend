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
