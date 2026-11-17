"""Atomic internal reservation and completion of idempotent test-chat turns."""

import secrets
from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.config import Settings
from dialogentis.models import Organization, User
from dialogentis.runtime.conversations import (
    TurnReservation,
    _locked_row,
    commit_turn,
    reserve_turn,
)
from dialogentis.runtime.core import RuntimeResult, TraceEvent
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.receipts import (
    ReceiptIdentity,
    ReceiptPayload,
    open_receipt,
    request_fingerprint,
    seal_receipt,
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


async def reserve_message(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    revision: int,
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
            created_by=user.id,
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
    user: User,
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
            RuntimeReceipt.created_by == user.id,
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
    if org.store_message_text:
        payload = ReceiptPayload.from_result(
            saved.revision, result.state.status, result.messages, result.trace, result.error_code
        )
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
