"""Transactional state operations for management conversations.

This internal service does not execute models or expose an API. Callers must run the
runtime with bounded execution and never return the decoded checkpoint to clients.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.config import Settings
from dialogentis.jobs.models import Job
from dialogentis.models import User
from dialogentis.organizations import organization_context
from dialogentis.runtime.checkpoint import (
    CheckpointIdentity,
    decrypt_checkpoint,
    encrypt_checkpoint,
)
from dialogentis.runtime.core import RuntimeState
from dialogentis.runtime.models import TestConversation
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import version_status


@dataclass(frozen=True)
class ConversationSnapshot:
    id: UUID
    revision: int
    state: RuntimeState = field(repr=False)


async def _access(db: AsyncSession, user: User, organization_id: UUID) -> None:
    if not user.email_verified:
        raise HTTPException(403, "Подтвердите адрес почты")
    await organization_context(db, user, organization_id, lock=True)


async def _now(db: AsyncSession) -> datetime:
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    return now


def _check_terminal(state: RuntimeState) -> None:
    if state.status != "waiting_input" and (
        state.slot_values or state.attempts or state.pending_ask
    ):
        raise ValueError("Terminal state must not contain runtime values")


def _identity(row: TestConversation) -> CheckpointIdentity:
    return CheckpointIdentity(row.organization_id, row.agent_id, row.version_id, row.id)


async def create_conversation(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    agent_id: UUID,
    version_id: UUID,
    state: RuntimeState,
    settings: Settings,
) -> ConversationSnapshot:
    await _access(db, user, organization_id)
    version_and_job = (
        await db.execute(
            select(AgentVersion, Job)
            .join(Job, Job.id == AgentVersion.job_id)
            .where(
                AgentVersion.organization_id == organization_id,
                AgentVersion.agent_id == agent_id,
                AgentVersion.id == version_id,
            )
        )
    ).one_or_none()
    if version_and_job is None:
        raise HTTPException(404, "Версия агента не найдена")
    version, job = version_and_job
    if version_status(version, job) != "ready":
        raise HTTPException(409, "Версия агента ещё не готова")
    _check_terminal(state)
    if state.version_id != version_id or state.turn != 0:
        raise ValueError("Invalid initial runtime state")
    now = await _now(db)
    row = TestConversation(
        id=uuid4(),
        organization_id=organization_id,
        agent_id=agent_id,
        version_id=version_id,
        created_by=user.id,
        revision=1,
        created_at=now,
        idle_expires_at=now + timedelta(minutes=30),
        absolute_expires_at=now + timedelta(hours=24),
    )
    row.checkpoint = (
        encrypt_checkpoint(state, _identity(row), settings)
        if state.status == "waiting_input"
        else None
    )
    db.add(row)
    await db.flush()
    return ConversationSnapshot(row.id, row.revision, state)


async def _locked_row(
    db: AsyncSession, user: User, organization_id: UUID, conversation_id: UUID
) -> tuple[TestConversation, datetime]:
    await _access(db, user, organization_id)
    row = await db.scalar(
        select(TestConversation)
        .where(
            TestConversation.organization_id == organization_id,
            TestConversation.id == conversation_id,
            TestConversation.created_by == user.id,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if row is None:
        raise HTTPException(404, "Диалог не найден")
    now = await _now(db)
    if row.idle_expires_at <= now or row.absolute_expires_at <= now:
        raise HTTPException(410, "Срок диалога истёк")
    if row.checkpoint is None:
        raise HTTPException(409, "Диалог завершён")
    return row, now


async def read_conversation(
    db: AsyncSession, user: User, organization_id: UUID, conversation_id: UUID, settings: Settings
) -> ConversationSnapshot:
    row, _ = await _locked_row(db, user, organization_id, conversation_id)
    assert row.checkpoint is not None
    state = decrypt_checkpoint(row.checkpoint, _identity(row), settings)
    return ConversationSnapshot(row.id, row.revision, state)


async def commit_turn(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    conversation_id: UUID,
    expected_revision: int,
    state: RuntimeState,
    settings: Settings,
) -> ConversationSnapshot:
    _check_terminal(state)
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("Invalid revision")
    row, now = await _locked_row(db, user, organization_id, conversation_id)
    if row.revision != expected_revision:
        raise HTTPException(409, "Диалог изменился. Обновите состояние")
    assert row.checkpoint is not None
    previous = decrypt_checkpoint(row.checkpoint, _identity(row), settings)
    if (
        state.version_id != previous.version_id
        or state.flow_hash != previous.flow_hash
        or state.turn != previous.turn + 1
    ):
        raise ValueError("Invalid runtime transition")
    row.checkpoint = (
        encrypt_checkpoint(state, _identity(row), settings)
        if state.status == "waiting_input"
        else None
    )
    row.revision += 1
    row.idle_expires_at = min(now + timedelta(minutes=30), row.absolute_expires_at)
    await db.flush()
    return ConversationSnapshot(row.id, row.revision, state)
