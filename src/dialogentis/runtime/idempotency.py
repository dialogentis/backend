"""Atomic internal reservation and completion of idempotent test-chat turns."""

import secrets
from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.channels.api_access import ApiAccess
from dialogentis.channels.widget_access import WidgetAccess
from dialogentis.config import Settings
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.models import Organization, User
from dialogentis.organizations import organization_context
from dialogentis.runtime.checkpoint import decrypt_checkpoint, encrypt_checkpoint
from dialogentis.runtime.conversations import (
    Principal,
    TurnReservation,
    _access,
    _check_terminal,
    _identity,
    _locked_row,
    commit_turn,
    create_conversation,
    created_by,
    reserve_turn,
)
from dialogentis.runtime.core import RuntimeResult, RuntimeState, TraceEvent
from dialogentis.runtime.history import record_history
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.receipts import (
    ReceiptIdentity,
    open_receipt,
    request_fingerprint,
    seal_receipt,
    start_identity,
)


@dataclass(frozen=True)
class Replay:
    turn_id: UUID
    revision: int
    status: str
    replay_available: bool
    messages: tuple[str, ...] = field(default=(), repr=False)
    trace: tuple[TraceEvent, ...] = field(default=(), repr=False)
    error_code: str | None = None


@dataclass(frozen=True)
class ReservedMessage:
    turn_id: UUID
    reservation: TurnReservation = field(repr=False)


@dataclass(frozen=True)
class ReservedStart:
    conversation_id: UUID
    credential: str = field(repr=False)
    turn: ReservedMessage = field(repr=False)


@dataclass(frozen=True)
class ReplayedStart:
    conversation_id: UUID
    credential: str = field(repr=False)
    result: Replay = field(repr=False)


async def reserve_initial_turn(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    agent_id: UUID,
    version_id: UUID,
    initial: RuntimeState,
    settings: Settings,
    *,
    idempotency_key: str | None = None,
    http_mode: str = "mock",
    release_id: UUID | None = None,
) -> ReservedStart | ReplayedStart:
    """Reserve a private initial turn; a caller must commit before executing it."""
    if (
        initial.version_id != version_id
        or initial.turn != 0
        or initial.status != "waiting_input"
        or initial.fallback_count != 0
        or initial.pending_ask is not None
        or initial.attempts
        or initial.active_path
        or initial.history_sources
        or initial.callers
    ):
        raise ValueError("Invalid initial state")
    key = idempotency_key if idempotency_key is not None else uuid4().hex
    conversation_id, message_id, credential = start_identity(
        settings, organization_id, user.id, agent_id, key
    )
    if isinstance(user, User) and http_mode == "live":
        await organization_context(db, user, organization_id, "manage_integrations", lock=True)
    else:
        await _access(db, user, organization_id)
    row = await db.get(TestConversation, conversation_id)
    receipt = await db.scalar(
        select(RuntimeReceipt.id).where(
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.conversation_id == conversation_id,
            RuntimeReceipt.client_message_id == message_id,
        )
    )
    if row is not None:
        if (
            row.agent_id != agent_id
            or row.created_by != created_by(user)
            or row.api_key_id != (user.id if isinstance(user, ApiAccess) else None)
            or row.widget_id != (user.id if isinstance(user, WidgetAccess) else None)
            or row.release_id != release_id
            or row.version_id != version_id
            or row.http_mode != http_mode
        ):
            raise HTTPException(409, "Ключ уже использован для другого диалога")
        if (row.revision != 1 and isinstance(user, User)) or receipt is None:
            raise HTTPException(409, "Диалог изменился или результат запуска недоступен")
        replay = await reserve_message(
            db,
            user,
            organization_id,
            conversation_id,
            credential,
            1,
            message_id,
            key,
            f"initial-turn:{version_id}",
            settings,
        )
        if not isinstance(replay, Replay):
            raise HTTPException(409, "Результат запуска недоступен")
        return ReplayedStart(conversation_id, credential, replay)
    if receipt is not None:
        raise HTTPException(409, "Срок диалога истёк; используйте новый ключ")
    created = await create_conversation(
        db,
        user,
        organization_id,
        agent_id,
        version_id,
        initial,
        settings,
        conversation_id=conversation_id,
        credential=credential,
        http_mode=http_mode,
        release_id=release_id,
    )
    assert created.credential is not None
    turn = await reserve_message(
        db,
        user,
        organization_id,
        created.id,
        created.credential,
        created.revision,
        message_id,
        key,
        f"initial-turn:{version_id}",
        settings,
    )
    assert isinstance(turn, ReservedMessage)
    return ReservedStart(created.id, created.credential, turn)


async def complete_initial_turn(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    reserved: ReservedStart,
    result: RuntimeResult,
    settings: Settings,
) -> Replay:
    """Commit the first result without counting it as a user message."""
    _check_terminal(result.state)
    row, now = await _locked_row(
        db, user, organization_id, reserved.conversation_id, reserved.credential
    )
    turn = reserved.turn
    receipt = await db.scalar(
        select(RuntimeReceipt)
        .where(
            RuntimeReceipt.id == turn.turn_id,
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.conversation_id == row.id,
            RuntimeReceipt.created_by == created_by(user),
        )
        .with_for_update()
    )
    if (
        row.id != turn.reservation.id
        or row.revision != turn.reservation.revision
        or row.processing_token != turn.reservation.token
        or row.processing_expires_at is None
        or row.processing_expires_at <= now
        or receipt is None
        or receipt.status != "processing"
        or receipt.processing_token != turn.reservation.token
    ):
        raise HTTPException(409, "Резервирование хода недействительно")
    if (
        await db.scalar(
            select(ActionExecution.id)
            .where(
                ActionExecution.organization_id == organization_id,
                ActionExecution.receipt_id == turn.turn_id,
                ActionExecution.status == "started",
            )
            .limit(1)
        )
        is not None
    ):
        raise HTTPException(409, "Внешнее действие ещё выполняется")
    assert row.checkpoint is not None
    previous = decrypt_checkpoint(row.checkpoint, _identity(row), settings)
    if (
        previous.turn != 0
        or result.state.turn != 0
        or result.state.version_id != previous.version_id
        or result.state.flow_hash != previous.flow_hash
    ):
        raise ValueError("Invalid initial runtime transition")
    row.checkpoint = (
        encrypt_checkpoint(result.state, _identity(row), settings)
        if result.state.status == "waiting_input"
        else None
    )
    row.processing_token = None
    row.processing_expires_at = None
    row.idle_expires_at = min(now + timedelta(minutes=30), row.absolute_expires_at)
    org = await db.get(Organization, organization_id)
    assert org is not None
    payload = await record_history(db, org, row, receipt.id, now, result, settings)
    if org.store_message_text:
        receipt.encrypted_response = seal_receipt(
            payload, ReceiptIdentity(organization_id, row.id, receipt.id), settings
        )
    receipt.status, receipt.revision, receipt.runtime_status = (
        "completed",
        row.revision,
        result.state.status,
    )
    if result.state.status != "waiting_input":
        receipt.expires_at = now + timedelta(hours=24)
    await db.flush()
    return Replay(
        receipt.id,
        row.revision,
        result.state.status,
        org.store_message_text,
        result.messages,
        result.trace,
        result.error_code,
    )


async def reserve_message(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    revision: int | None,
    message_id: UUID,
    idempotency_key: str,
    text: str,
    settings: Settings,
) -> ReservedMessage | Replay:
    if not isinstance(message_id, UUID):
        raise ValueError("Invalid message ID")
    key_hash, request_digest = request_fingerprint(
        settings, organization_id, conversation_id, message_id, idempotency_key, text
    )
    row, now = await _locked_row(
        db, user, organization_id, conversation_id, credential, allow_terminal=True
    )
    if isinstance(user, User) and row.http_mode == "live":
        await organization_context(db, user, organization_id, "manage_integrations", lock=True)
    matches = (
        await db.scalars(
            select(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
                or_(
                    RuntimeReceipt.client_message_id == message_id,
                    RuntimeReceipt.idempotency_key_hash == key_hash,
                ),
            )
            .with_for_update()
        )
    ).all()
    if matches:
        if len(matches) != 1:
            raise HTTPException(409, "Идентификаторы уже использованы в разных ходах")
        item = matches[0]
        if (
            item.client_message_id != message_id
            or item.idempotency_key_hash != key_hash
            or item.key_version != settings.encryption_key_version
            or not secrets.compare_digest(item.request_digest, request_digest)
        ):
            raise HTTPException(409, "Идентификатор уже использован с другим запросом")
        if item.status == "processing":
            raise HTTPException(409, "Ход уже обрабатывается", headers={"Retry-After": "1"})
        if item.status != "completed":
            raise HTTPException(409, "Результат хода неизвестен; начните диалог заново")
        org = await db.get(Organization, organization_id)
        assert org is not None and item.revision is not None and item.runtime_status is not None
        if org.store_message_text and item.encrypted_response is not None:
            content = open_receipt(
                item.encrypted_response,
                ReceiptIdentity(organization_id, conversation_id, item.id),
                settings,
            )
            if content.revision != item.revision or content.status != item.runtime_status:
                raise ValueError("Receipt metadata mismatch")
            trace = tuple(TraceEvent(**event.model_dump()) for event in content.trace)
            return Replay(
                item.id,
                item.revision,
                item.runtime_status,
                True,
                content.messages,
                trace,
                content.error_code,
            )
        return Replay(item.id, item.revision, item.runtime_status, False)
    if row.checkpoint is None:
        raise HTTPException(409, "Диалог завершён")
    reservation = await reserve_turn(
        db, user, organization_id, conversation_id, revision, settings, credential
    )
    assert row.credential_hash is not None
    turn_id = uuid4()
    db.add(
        RuntimeReceipt(
            id=turn_id,
            organization_id=organization_id,
            conversation_id=conversation_id,
            created_by=created_by(user),
            credential_hash=row.credential_hash,
            client_message_id=message_id,
            idempotency_key_hash=key_hash,
            request_digest=request_digest,
            key_version=settings.encryption_key_version,
            processing_token=reservation.token,
            status="processing",
            expires_at=row.absolute_expires_at + timedelta(hours=24),
        )
    )
    await db.flush()
    return ReservedMessage(turn_id, reservation)


async def complete_message(
    db: AsyncSession,
    user: Principal,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    reserved: ReservedMessage,
    result: RuntimeResult,
    settings: Settings,
) -> Replay:
    row, now = await _locked_row(db, user, organization_id, conversation_id, credential)
    item = await db.scalar(
        select(RuntimeReceipt)
        .where(
            RuntimeReceipt.id == reserved.turn_id,
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.conversation_id == conversation_id,
            RuntimeReceipt.created_by == created_by(user),
        )
        .with_for_update()
    )
    if (
        item is None
        or item.status != "processing"
        or item.processing_token != reserved.reservation.token
        or row.processing_token != reserved.reservation.token
    ):
        raise HTTPException(409, "Резервирование хода недействительно")
    active_action = await db.scalar(
        select(ActionExecution.id)
        .where(
            ActionExecution.organization_id == organization_id,
            ActionExecution.receipt_id == reserved.turn_id,
            ActionExecution.status == "started",
        )
        .limit(1)
    )
    if active_action is not None:
        raise HTTPException(409, "Внешнее действие ещё выполняется")
    saved = await commit_turn(
        db,
        user,
        organization_id,
        conversation_id,
        reserved.reservation.revision,
        reserved.reservation.token,
        result.state,
        settings,
        credential,
    )
    org = await db.get(Organization, organization_id)
    assert org is not None
    payload = await record_history(db, org, row, item.id, now, result, settings)
    if org.store_message_text:
        item.encrypted_response = seal_receipt(
            payload, ReceiptIdentity(organization_id, conversation_id, item.id), settings
        )
    item.status, item.revision, item.runtime_status = (
        "completed",
        saved.revision,
        result.state.status,
    )
    if result.state.status != "waiting_input":
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
            )
            .values(expires_at=now + timedelta(hours=24))
        )
    await db.flush()
    return Replay(
        item.id,
        saved.revision,
        result.state.status,
        org.store_message_text,
        result.messages,
        result.trace,
        result.error_code,
    )
