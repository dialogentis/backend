"""Authenticated management routes for testing a ready agent version."""

from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt, StrictStr

from dialogentis.api_errors import error_response
from dialogentis.auth.security import current_user
from dialogentis.contracts import InputModel
from dialogentis.datasets.routes import issue_view
from dialogentis.db import unit_of_work
from dialogentis.runtime.executor import EventView, ExecutionError
from dialogentis.runtime.test_chat import ChatResponse, InvalidVersionFlow, TestChatService

router = APIRouter(prefix="/api/v1/organizations/{org_id}", tags=["test-chat"])
ConversationToken = Annotated[
    str, Header(alias="X-Conversation-Token", pattern=r"^[A-Za-z0-9_-]{43}$")
]
IdempotencyKey = Annotated[
    str, Header(alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[!-~]+$")
]


async def chat_user_id(request: Request) -> UUID:
    factory = request.app.state.session_factory
    if factory is None:
        raise HTTPException(503, "База данных недоступна")
    async with unit_of_work(factory) as db:
        user = await current_user(request, db)
        if not user.email_verified:
            raise HTTPException(403, "Подтвердите адрес электронной почты")
        return user.id


ChatUser = Annotated[UUID, Depends(chat_user_id)]


class StartInput(InputModel):
    version_id: UUID
    http_mode: Literal["mock", "live"] = "mock"


class MessageInput(InputModel):
    client_message_id: UUID
    revision: StrictInt = Field(ge=1)
    text: StrictStr = Field(min_length=1, max_length=2000, repr=False)


class ChatView(BaseModel):
    conversation_id: UUID
    revision: int
    status: Literal["waiting_input", "completed", "failed"]
    messages: tuple[str, ...] = Field(repr=False)
    trace: tuple[EventView, ...]
    error_code: str | None
    turn_id: UUID | None
    replayed: bool
    replay_available: bool


class StartedChat(ChatView):
    conversation_token: str = Field(repr=False)


def service(request: Request) -> TestChatService:
    chat = request.app.state.test_chat_service
    if chat is None:
        raise HTTPException(503, "Тестовый чат временно недоступен")
    return chat


def view(result: ChatResponse) -> ChatView:
    return ChatView(
        conversation_id=result.conversation_id,
        revision=result.revision,
        status=cast(Literal["waiting_input", "completed", "failed"], result.status),
        messages=result.messages,
        trace=tuple(EventView(**vars(event)) for event in result.trace),
        error_code=result.error_code,
        turn_id=result.turn_id,
        replayed=result.replayed,
        replay_available=result.replay_available,
    )


def invalid_flow_response(request: Request, exc: InvalidVersionFlow):
    return error_response(
        request,
        409,
        "invalid_version_flow",
        "В сценарии этой версии есть ошибки. Исправьте черновик и подготовьте новую версию.",
        [issue_view(issue).model_dump(mode="json") for issue in exc.errors[:100]],
    )


@router.post("/agents/{agent_id}/test-conversations", response_model=StartedChat, status_code=201)
async def start_chat(
    org_id: UUID,
    agent_id: UUID,
    body: StartInput,
    request: Request,
    user_id: ChatUser,
    key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[!-~]+$"),
    ] = None,
):
    try:
        result = await service(request).start(
            user_id,
            org_id,
            agent_id,
            body.version_id,
            idempotency_key=key,
            http_mode=body.http_mode,
        )
    except InvalidVersionFlow as exc:
        return invalid_flow_response(request, exc)
    except ExecutionError:
        raise HTTPException(503, "Не удалось запустить тестовый диалог") from None
    assert result.credential is not None
    return StartedChat(**view(result).model_dump(), conversation_token=result.credential)


@router.post("/test-conversations/{conversation_id}/messages", response_model=ChatView)
async def send_message(
    org_id: UUID,
    conversation_id: UUID,
    body: MessageInput,
    request: Request,
    user_id: ChatUser,
    token: ConversationToken,
    key: IdempotencyKey,
):
    try:
        result = await service(request).send(
            user_id,
            org_id,
            conversation_id,
            body.revision,
            body.text,
            token,
            body.client_message_id,
            key,
        )
    except InvalidVersionFlow as exc:
        return invalid_flow_response(request, exc)
    except ExecutionError:
        raise HTTPException(503, "Не удалось обработать сообщение") from None
    return view(result)
