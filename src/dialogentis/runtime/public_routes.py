from datetime import datetime, timedelta
from typing import Annotated, Literal, cast
from uuid import UUID, uuid5

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, StrictStr
from sqlalchemy import select, update

from dialogentis.artifact_service import ArtifactUnavailable
from dialogentis.channels.api_access import ApiAccess, identify_api_key, invalid_key
from dialogentis.channels.api_key_models import KeyScope
from dialogentis.contracts import InputModel
from dialogentis.db import unit_of_work
from dialogentis.runtime.conversations import _locked_row
from dialogentis.runtime.executor import ExecutionError
from dialogentis.runtime.history_models import ConversationHistory
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.runtime.routes import ConversationToken, IdempotencyKey, service
from dialogentis.runtime.test_chat import ChatResponse, InvalidVersionFlow

router = APIRouter(prefix="/api/v1/runtime", tags=["runtime"])
Bearer = Annotated[HTTPAuthorizationCredentials | None, Security(HTTPBearer(auto_error=False))]
type RuntimeStatus = Literal["waiting_input", "completed", "failed"]


def require_scope(scope: KeyScope):
    async def access(request: Request, credential: Bearer) -> ApiAccess:
        if credential is None:
            raise invalid_key()
        factory = request.app.state.session_factory
        if factory is None:
            raise HTTPException(503, "Диалоги временно недоступны")
        async with unit_of_work(factory) as db:
            return await identify_api_key(db, credential.credentials, scope)

    return access


StartAccess = Annotated[ApiAccess, Depends(require_scope("conversation:create"), scope="function")]
MessageAccess = Annotated[ApiAccess, Depends(require_scope("message:send"), scope="function")]


class RuntimeMessageInput(InputModel):
    client_message_id: UUID
    text: StrictStr = Field(min_length=1, max_length=2000, repr=False)


class TextMessage(BaseModel):
    id: UUID
    type: Literal["text"] = "text"
    text: str = Field(repr=False)


class MessageView(BaseModel):
    conversation_id: UUID
    turn_id: UUID
    sequence: int
    status: RuntimeStatus
    messages: list[TextMessage]
    replayed: bool
    replay_available: bool
    request_id: str


class StartedConversation(MessageView):
    conversation_token: str = Field(repr=False)
    release_id: UUID
    expires_at: datetime


class ReceiptView(BaseModel):
    turn_id: UUID
    status: Literal["processing", "completed", "unknown"]
    sequence: int | None


def view(request: Request, result: ChatResponse) -> MessageView:
    assert result.turn_id is not None
    return MessageView(
        conversation_id=result.conversation_id,
        turn_id=result.turn_id,
        sequence=result.revision - 1,
        status=cast(RuntimeStatus, result.status),
        messages=[
            TextMessage(id=uuid5(result.turn_id, str(index)), text=value)
            for index, value in enumerate(result.messages)
        ],
        replayed=result.replayed,
        replay_available=result.replay_available,
        request_id=request.state.request_id,
    )


@router.post(
    "/agents/{agent_id}/conversations", response_model=StartedConversation, status_code=201
)
async def start(
    agent_id: UUID, request: Request, access: StartAccess, key: IdempotencyKey
) -> StartedConversation:
    if agent_id != access.agent_id:
        raise HTTPException(404, "Агент недоступен")
    try:
        result = await service(request).start_public(access, key)
    except ArtifactUnavailable, ExecutionError, InvalidVersionFlow:
        raise HTTPException(503, "Не удалось запустить диалог") from None
    assert result.release_id is not None and result.expires_at is not None and result.credential
    return StartedConversation(
        **view(request, result).model_dump(),
        conversation_token=result.credential,
        release_id=result.release_id,
        expires_at=result.expires_at,
    )


@router.post("/conversations/{conversation_id}/messages", response_model=MessageView)
async def send(
    conversation_id: UUID,
    body: RuntimeMessageInput,
    request: Request,
    access: MessageAccess,
    token: ConversationToken,
    key: IdempotencyKey,
) -> MessageView:
    try:
        result = await service(request).send(
            access,
            access.organization_id,
            conversation_id,
            None,
            body.text,
            token,
            body.client_message_id,
            key,
        )
    except ArtifactUnavailable, ExecutionError, InvalidVersionFlow:
        raise HTTPException(503, "Не удалось обработать сообщение") from None
    return view(request, result)


@router.post("/conversations/{conversation_id}/close", status_code=204)
async def close(
    conversation_id: UUID, request: Request, access: MessageAccess, token: ConversationToken
) -> None:
    async with unit_of_work(
        request.app.state.session_factory, organization_id=access.organization_id
    ) as db:
        await access.authorize(db)
        row, now = await _locked_row(
            db, access, access.organization_id, conversation_id, token, allow_terminal=True
        )
        if row.processing_token is not None:
            raise HTTPException(409, "Ход уже обрабатывается", headers={"Retry-After": "1"})
        row.checkpoint = None
        history = await db.get(ConversationHistory, (access.organization_id, conversation_id))
        if history is not None and history.status == "waiting_input":
            history.status = "completed"
            history.last_activity_at = now
        await db.execute(
            update(RuntimeReceipt)
            .where(
                RuntimeReceipt.organization_id == access.organization_id,
                RuntimeReceipt.conversation_id == conversation_id,
                RuntimeReceipt.expires_at > now + timedelta(hours=24),
            )
            .values(expires_at=now + timedelta(hours=24))
        )


@router.get(
    "/conversations/{conversation_id}/receipt/{client_message_id}", response_model=ReceiptView
)
async def receipt(
    conversation_id: UUID,
    client_message_id: UUID,
    request: Request,
    access: MessageAccess,
    token: ConversationToken,
) -> ReceiptView:
    async with unit_of_work(
        request.app.state.session_factory, organization_id=access.organization_id
    ) as db:
        await access.authorize(db)
        await _locked_row(
            db, access, access.organization_id, conversation_id, token, allow_terminal=True
        )
        item = (
            await db.execute(
                select(RuntimeReceipt.id, RuntimeReceipt.status, RuntimeReceipt.revision).where(
                    RuntimeReceipt.organization_id == access.organization_id,
                    RuntimeReceipt.conversation_id == conversation_id,
                    RuntimeReceipt.client_message_id == client_message_id,
                )
            )
        ).one_or_none()
        if item is None:
            raise HTTPException(404, "Результат хода не найден")
        return ReceiptView(
            turn_id=item.id,
            status=cast(Literal["processing", "completed", "unknown"], item.status),
            sequence=None if item.revision is None else item.revision - 1,
        )
