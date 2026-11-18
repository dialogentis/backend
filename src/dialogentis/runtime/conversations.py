"""Transactional state operations for management conversations.

This internal service does not execute models or expose an API. Callers must run the
runtime with bounded execution and never return the decoded checkpoint to clients.
"""

import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.auth.security import digest
from dialogentis.channels.api_access import ApiAccess
from dialogentis.channels.widget_access import WidgetAccess
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
from dialogentis.runtime.history import start_history
from dialogentis.runtime.models import TestConversation
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import version_status

PublicAccess = ApiAccess | WidgetAccess
type Principal = User | PublicAccess


def created_by(principal: Principal) -> UUID | None:
    return principal.id if isinstance(principal, User) else None


@dataclass(frozen=True)
class ConversationSnapshot:
    id: UUID
    revision: int
    state: RuntimeState = field(repr=False)
    credential: str | None = field(default=None, repr=False)


async def _access(db: AsyncSession, user: Principal, organization_id: UUID) -> None:
    if isinstance(user, PublicAccess):
        await user.lock_organization(db, organization_id)
        return
    if not user.email_verified:
        raise HTTPException(403, "Подтвердите адрес почты")
    await organization_context(db, user, organization_id, lock=True)


async def _now(db: AsyncSession) -> datetime:
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    return now


def _check_terminal(state: RuntimeState) -> None:
    if state.status != "waiting_input" and (
        state.slot_values
        or state.attempts
        or state.pending_ask
        or state.active_path
        or state.history_sources
        or state.callers
    ):
        raise ValueError("Terminal state must not contain runtime values")


def _identity(row: TestConversation) -> CheckpointIdentity:
    return CheckpointIdentity(row.organization_id, row.agent_id, row.version_id, row.id)


async def create_conversation(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    agent_id: UUID,
    version_id: UUID,
    state: RuntimeState,
    settings: Settings,
    *,
    conversation_id: UUID | None = None,
    credential: str | None = None,
    http_mode: str = "mock",
    release_id: UUID | None = None,
) -> ConversationSnapshot:
    await _access(db, user, organization_id)
    if isinstance(user, PublicAccess) and (agent_id != user.agent_id or release_id is None):
        raise HTTPException(404, "Агент недоступен")
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
    if (conversation_id is None) != (credential is None):
        raise ValueError("Invalid conversation identity")
    if credential is not None and re.fullmatch(r"[A-Za-z0-9_-]{43}", credential) is None:
        raise ValueError("Invalid conversation identity")
    credential = credential or secrets.token_urlsafe(32)
    row = TestConversation(
        id=conversation_id or uuid4(),
        organization_id=organization_id,
        agent_id=agent_id,
        version_id=version_id,
        created_by=created_by(user),
        api_key_id=user.id if isinstance(user, ApiAccess) else None,
        widget_id=user.id if isinstance(user, WidgetAccess) else None,
        release_id=release_id,
        credential_hash=digest(credential),
        http_mode=http_mode,
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
    start_history(db, row, status=state.status)
    await db.flush()
    return ConversationSnapshot(row.id, row.revision, state, credential)


async def _locked_row(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    *,
    allow_terminal: bool = False,
) -> tuple[TestConversation, datetime]:
    await _access(db, user, organization_id)
    row = await db.scalar(
        select(TestConversation)
        .where(
            TestConversation.organization_id == organization_id,
            TestConversation.id == conversation_id,
            TestConversation.created_by == created_by(user),
            TestConversation.api_key_id == (user.id if isinstance(user, ApiAccess) else None),
            TestConversation.widget_id == (user.id if isinstance(user, WidgetAccess) else None),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if (
        row is None
        or not isinstance(credential, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{43}", credential) is None
        or row.credential_hash is None
        or not secrets.compare_digest(row.credential_hash, digest(credential))
    ):
        raise HTTPException(404, "Диалог не найден")
    now = await _now(db)
    if row.idle_expires_at <= now or row.absolute_expires_at <= now:
        raise HTTPException(410, "Срок диалога истёк")
    if row.checkpoint is None and not allow_terminal:
        raise HTTPException(409, "Диалог завершён")
    return row, now


@dataclass(frozen=True)
class TurnReservation:
    id: UUID
    revision: int
    token: UUID = field(repr=False)
    state: RuntimeState = field(repr=False)


async def reserve_turn(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    expected_revision: int | None,
    settings: Settings,
    credential: str,
) -> TurnReservation:
    if not (
        (type(expected_revision) is int and expected_revision >= 1)
        or (expected_revision is None and isinstance(user, PublicAccess))
    ):
        raise ValueError("Invalid revision")
    row, now = await _locked_row(db, user, organization_id, conversation_id, credential)
    if row.processing_token is not None:
        raise HTTPException(
            409, "Ход уже обрабатывается или требует восстановления", headers={"Retry-After": "1"}
        )
    if expected_revision is not None and row.revision != expected_revision:
        raise HTTPException(409, "Диалог изменился. Обновите состояние")
    assert row.checkpoint is not None
    state = decrypt_checkpoint(row.checkpoint, _identity(row), settings)
    token = uuid4()
    row.processing_token = token
    row.processing_expires_at = min(
        now + timedelta(seconds=30), row.absolute_expires_at, row.idle_expires_at
    )
    await db.flush()
    return TurnReservation(row.id, row.revision, token, state)


async def read_conversation(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    settings: Settings,
    credential: str,
) -> ConversationSnapshot:
    row, _ = await _locked_row(db, user, organization_id, conversation_id, credential)
    if row.processing_token is not None:
        raise HTTPException(
            409, "Ход уже обрабатывается или требует восстановления", headers={"Retry-After": "1"}
        )
    assert row.checkpoint is not None
    state = decrypt_checkpoint(row.checkpoint, _identity(row), settings)
    return ConversationSnapshot(row.id, row.revision, state)


async def commit_turn(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    expected_revision: int,
    token: UUID,
    state: RuntimeState,
    settings: Settings,
    credential: str,
) -> ConversationSnapshot:
    _check_terminal(state)
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("Invalid revision")
    row, now = await _locked_row(db, user, organization_id, conversation_id, credential)
    if row.revision != expected_revision:
        raise HTTPException(409, "Диалог изменился. Обновите состояние")
    if (
        row.processing_token != token
        or row.processing_expires_at is None
        or row.processing_expires_at <= now
    ):
        raise HTTPException(409, "Резервирование хода недействительно")
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
    row.processing_token = None
    row.processing_expires_at = None
    row.revision += 1
    row.idle_expires_at = min(now + timedelta(minutes=30), row.absolute_expires_at)
    await db.flush()
    return ConversationSnapshot(row.id, row.revision, state)
