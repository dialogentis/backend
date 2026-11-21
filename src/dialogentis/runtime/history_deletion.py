import asyncio
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import Uuid, bindparam, delete, func, select, update
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.config import get_settings
from dialogentis.db import unit_of_work
from dialogentis.deletion_journal import HistoryMarker, write_history_marker
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError, RetryableJobError
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


async def expire_history_exports(db: AsyncSession, organization_id: UUID) -> None:
    artifacts = select(HistoryExportTask.artifact_id).where(
        HistoryExportTask.organization_id == organization_id,
        HistoryExportTask.artifact_id.is_not(None),
    )
    await db.execute(
        update(Artifact)
        .where(
            Artifact.organization_id == organization_id,
            Artifact.id.in_(artifacts),
            Artifact.expires_at > func.clock_timestamp(),
        )
        .values(expires_at=func.clock_timestamp())
    )


async def apply_history_marker(
    db: AsyncSession, org: Organization, marker: HistoryMarker
) -> tuple[int, int, int]:
    targets = select(
        func.unnest(bindparam("conversation_ids", marker.conversation_ids, type_=ARRAY(Uuid())))
    )
    later_turns = select(HistoryTurn.turn_id).where(
        HistoryTurn.organization_id == org.id,
        HistoryTurn.conversation_id.in_(targets),
        HistoryTurn.created_at > marker.requested_at,
    )
    # Keep the idempotency record and external-action ledger after removing its payload.
    cleared_receipts = await count_changes(
        db,
        update(RuntimeReceipt)
        .where(
            RuntimeReceipt.organization_id == org.id,
            RuntimeReceipt.conversation_id.in_(targets),
            RuntimeReceipt.created_at <= marker.requested_at,
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
            HistoryTurn.created_at <= marker.requested_at,
        )
        .returning(HistoryTurn.turn_id),
    )
    deleted_dialogues = await count_changes(
        db,
        delete(ConversationHistory)
        .where(
            ConversationHistory.organization_id == org.id,
            ConversationHistory.id.in_(targets),
            ConversationHistory.last_activity_at <= marker.requested_at,
        )
        .returning(ConversationHistory.id),
    )
    if deleted_dialogues or deleted_turns or cleared_receipts:
        org.settings_revision += 1
        await expire_history_exports(db, org.id)
    return deleted_dialogues, deleted_turns, cleared_receipts


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
        conversation_ids = list(
            await db.scalars(
                select(HistoryDeletionTarget.conversation_id)
                .where(*matches)
                .order_by(HistoryDeletionTarget.conversation_id)
            )
        )
        deleted_dialogues = deleted_turns = cleared_receipts = 0
        marker = None
        if conversation_ids:
            marker = HistoryMarker(
                organization_id=org.id,
                job_id=job.id,
                requested_at=task.requested_at,
                conversation_ids=conversation_ids,
            )
            deleted_dialogues, deleted_turns, cleared_receipts = await apply_history_marker(
                db, org, marker
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
        if marker is not None:
            await db.flush()
            try:
                # Keep the authorized deletion across restores of a snapshot taken before commit.
                await asyncio.to_thread(
                    write_history_marker, get_settings().deletion_journal_root, marker
                )
            except OSError as exc:
                raise RetryableJobError("Deletion journal unavailable") from exc

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
