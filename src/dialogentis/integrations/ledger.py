"""Reserve and fence external actions before any network call.

The caller must commit the reservation before sending; these internal functions
do not authorize a graph node or perform network operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, get_args
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.integrations.models import Integration
from dialogentis.models import User
from dialogentis.runtime.conversations import _locked_row
from dialogentis.runtime.idempotency import ReservedMessage
from dialogentis.runtime.receipt_models import RuntimeReceipt

Outcome = Literal[
    "succeeded",
    "http_error",
    "timeout",
    "invalid_response",
    "unknown_result",
    "network_error",
    "address_blocked",
    "request_not_allowed",
    "dns_error",
    "invalid_request",
    "invalid_credential",
]
_OUTCOMES = frozenset(get_args(Outcome))


@dataclass(frozen=True)
class ActionReservation:
    id: UUID
    receipt_id: UUID
    conversation_id: UUID
    integration_id: UUID
    node_id: UUID
    ordinal: int
    token: UUID = field(repr=False)


async def _active_turn(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    reserved: ReservedMessage,
) -> datetime:
    row, now = await _locked_row(db, user, organization_id, conversation_id, credential)
    receipt = await db.scalar(
        select(RuntimeReceipt)
        .where(
            RuntimeReceipt.organization_id == organization_id,
            RuntimeReceipt.id == reserved.turn_id,
            RuntimeReceipt.conversation_id == conversation_id,
            RuntimeReceipt.created_by == user.id,
        )
        .with_for_update()
    )
    if (
        reserved.reservation.id != conversation_id
        or row.processing_token != reserved.reservation.token
        or row.revision != reserved.reservation.revision
        or row.processing_expires_at is None
        or row.processing_expires_at <= now
        or receipt is None
        or receipt.status != "processing"
        or receipt.processing_token != reserved.reservation.token
    ):
        raise HTTPException(409, "Резервирование хода недействительно")
    return now


async def begin_action(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    conversation_id: UUID,
    credential: str,
    reserved: ReservedMessage,
    integration_id: UUID,
    node_id: UUID,
    ordinal: int,
    method: str,
) -> ActionReservation:
    if (
        not isinstance(integration_id, UUID)
        or not isinstance(node_id, UUID)
        or type(ordinal) is not int
        or not 1 <= ordinal <= 3
        or not isinstance(method, str)
        or method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}
    ):
        raise ValueError("Invalid action")
    await _active_turn(db, user, organization_id, conversation_id, credential, reserved)
    integration = await db.scalar(
        select(Integration).where(
            Integration.organization_id == organization_id,
            Integration.id == integration_id,
        )
    )
    if integration is None:
        raise HTTPException(404, "Интеграция не найдена")
    if not integration.enabled or method not in integration.methods:
        raise HTTPException(409, "Интеграция отключена или метод недоступен")
    previous = (
        await db.scalars(
            select(ActionExecution)
            .where(
                ActionExecution.organization_id == organization_id,
                ActionExecution.receipt_id == reserved.turn_id,
            )
            .order_by(ActionExecution.ordinal)
        )
    ).all()
    if ordinal != len(previous) + 1 or any(item.status == "started" for item in previous):
        raise HTTPException(409, "Действие уже запущено или нарушен порядок")
    item = ActionExecution(
        id=uuid4(),
        organization_id=organization_id,
        conversation_id=conversation_id,
        receipt_id=reserved.turn_id,
        integration_id=integration_id,
        node_id=node_id,
        processing_token=reserved.reservation.token,
        ordinal=ordinal,
        method=method,
        status="started",
    )
    db.add(item)
    await db.flush()
    return ActionReservation(
        item.id,
        item.receipt_id,
        conversation_id,
        integration_id,
        node_id,
        ordinal,
        item.processing_token,
    )


async def finish_action(
    db: AsyncSession,
    user: User,
    organization_id: UUID,
    credential: str,
    turn: ReservedMessage,
    action: ActionReservation,
    outcome: Outcome,
    http_status: int | None = None,
) -> None:
    if (
        not isinstance(outcome, str)
        or outcome not in _OUTCOMES
        or (
            outcome == "succeeded"
            and (type(http_status) is not int or not 200 <= http_status <= 299)
        )
        or (outcome != "succeeded" and http_status is not None)
    ):
        raise ValueError("Invalid action outcome")
    now = await _active_turn(db, user, organization_id, action.conversation_id, credential, turn)
    item = await db.scalar(
        select(ActionExecution)
        .where(
            ActionExecution.id == action.id,
            ActionExecution.organization_id == organization_id,
            ActionExecution.receipt_id == turn.turn_id,
        )
        .with_for_update()
    )
    if (
        item is None
        or item.status != "started"
        or item.processing_token != turn.reservation.token
        or action.token != turn.reservation.token
        or item.conversation_id != action.conversation_id
        or item.integration_id != action.integration_id
        or item.node_id != action.node_id
        or item.ordinal != action.ordinal
        or action.receipt_id != turn.turn_id
    ):
        raise HTTPException(409, "Резервирование действия недействительно")
    item.status, item.http_status, item.finished_at = outcome, http_status, now
    await db.flush()
