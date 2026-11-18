from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError
from dialogentis.jobs.models import Job
from dialogentis.models import Organization, User
from dialogentis.organizations import audit, organization_context
from dialogentis.runtime.history_deletion_models import HistoryDeletionTarget, HistoryDeletionTask
from dialogentis.runtime.history_export_models import HistoryExportTask
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.receipt_models import RuntimeReceipt


async def count_changes(db: AsyncSession, mutation: Any) -> int:
    return int(await db.scalar(select(func.count()).select_from(mutation.cte("changed"))) or 0)


async def deletion_actor(db: AsyncSession, task: HistoryDeletionTask) -> User:
    user = await db.get(User, task.requested_by)
    if user is None or not user.email_verified:
        raise PermanentJobError("history_deletion_access_changed")
    try:
        await organization_context(db, user, task.organization_id, "manage_history")
    except HTTPException:
        raise PermanentJobError("history_deletion_access_changed") from None
    return user


async def prepare_history_deletion(
    factory: async_sessionmaker[AsyncSession], current: Claim
) -> Apply:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        task = await db.get(HistoryDeletionTask, current.job_id)
        if task is None:
            raise PermanentJobError("history_deletion_unavailable")
        await deletion_actor(db, task)

    async def apply(db: AsyncSession, job: Job) -> None:
        task = await db.get(HistoryDeletionTask, job.id)
        if task is None:
            raise PermanentJobError("history_deletion_unavailable")
        user = await deletion_actor(db, task)
        org = await db.get(Organization, task.organization_id)
        assert org is not None
        matches = (
            HistoryDeletionTarget.organization_id == org.id,
            HistoryDeletionTarget.job_id == job.id,
        )
        targets = select(HistoryDeletionTarget.conversation_id).where(*matches)
        later_turns = select(HistoryTurn.turn_id).where(
            HistoryTurn.organization_id == org.id,
            HistoryTurn.conversation_id.in_(targets),
            HistoryTurn.created_at > task.requested_at,
        )
        # Keep the idempotency record and external-action ledger after removing its payload.
        cleared_receipts = await count_changes(
            db,
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == org.id,
                RuntimeReceipt.conversation_id.in_(targets),
                RuntimeReceipt.created_at <= task.requested_at,
                RuntimeReceipt.encrypted_response.is_not(None),
                RuntimeReceipt.id.not_in(later_turns),
            )
            .values(encrypted_response=None)
            .returning(RuntimeReceipt.id),
        )
        deleted_turns = await count_changes(
            db,
            delete(HistoryTurn)
            .where(
                HistoryTurn.organization_id == org.id,
                HistoryTurn.conversation_id.in_(targets),
                HistoryTurn.created_at <= task.requested_at,
            )
            .returning(HistoryTurn.turn_id),
        )
        deleted_dialogues = await count_changes(
            db,
            delete(ConversationHistory)
            .where(
                ConversationHistory.organization_id == org.id,
                ConversationHistory.id.in_(targets),
                ConversationHistory.last_activity_at <= task.requested_at,
            )
            .returning(ConversationHistory.id),
        )
        if deleted_dialogues or deleted_turns or cleared_receipts:
            org.settings_revision += 1
            artifacts = select(HistoryExportTask.artifact_id).where(
                HistoryExportTask.organization_id == org.id,
                HistoryExportTask.artifact_id.is_not(None),
            )
            await db.execute(
                update(Artifact)
                .where(
                    Artifact.organization_id == org.id,
                    Artifact.id.in_(artifacts),
                    Artifact.expires_at > func.clock_timestamp(),
                )
                .values(expires_at=func.clock_timestamp())
            )
        task.deleted_dialogues = deleted_dialogues
        task.deleted_turns = deleted_turns
        task.cleared_receipts = cleared_receipts
        await db.execute(delete(HistoryDeletionTarget).where(*matches))
        audit(
            db,
            org.id,
            user,
            "history.deleted",
            job.id,
            details={
                "dialogues": deleted_dialogues,
                "turns": deleted_turns,
                "cleared_receipts": cleared_receipts,
            },
        )

    return apply


async def scrub_history_deletions(db: AsyncSession, organization_id: UUID) -> None:
    terminal = select(Job.id).where(
        Job.organization_id == organization_id,
        Job.status.in_(("succeeded", "failed", "cancelled")),
        Job.kind == "history.deletion",
    )
    await db.execute(
        delete(HistoryDeletionTarget).where(
            HistoryDeletionTarget.organization_id == organization_id,
            HistoryDeletionTarget.job_id.in_(terminal),
        )
    )
