from html import escape
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import Field

from dialogentis.artifact_service import ArtifactUnavailable
from dialogentis.channels.widget_access import (
    WidgetAccess,
    WidgetGrant,
    issue_widget_token,
    lookup_widget,
    read_widget_token,
    unavailable,
)
from dialogentis.channels.widget_models import WidgetChannel
from dialogentis.contracts import InputModel
from dialogentis.db import unit_of_work
from dialogentis.runtime.executor import ExecutionError
from dialogentis.runtime.public_routes import Bearer, MessageView, RuntimeMessageInput, view
from dialogentis.runtime.routes import IdempotencyKey, service
from dialogentis.runtime.test_chat import InvalidVersionFlow

router = APIRouter(prefix="/api/v1/widget", tags=["widget"])


class SessionInput(InputModel):
    parent_origin: str = Field(min_length=1, max_length=512)


class WidgetSession(MessageView):
    session_token: str = Field(repr=False)


async def channel_for(request: Request, channel_id: UUID) -> WidgetChannel:
    factory = request.app.state.session_factory
    if factory is None:
        raise unavailable(503)
    async with unit_of_work(factory) as db:
        channel = await lookup_widget(db, channel_id)
        if channel is None:
            raise unavailable()
        return channel


def require_frame_origin(request: Request) -> None:
    base = urlsplit(request.app.state.settings.public_base_url)
    if request.headers.get("origin") != f"{base.scheme}://{base.netloc}":
        raise unavailable(403)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/channels/{channel_id}/frame", response_class=HTMLResponse)
async def frame(channel_id: UUID, request: Request) -> HTMLResponse:
    channel = await channel_for(request, channel_id)
    name = escape(channel.name)
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{name}</title>
  <link rel="stylesheet" href="/widget-frame.css">
  <script src="/widget-frame.js" defer></script>
</head>
<body data-channel="{channel_id}" data-position="{escape(channel.position)}"
      data-accent="{escape(channel.accent_color)}" data-enabled="{str(channel.enabled).lower()}">
  <button type="button" id="open-chat" aria-controls="chat" aria-expanded="false" hidden>
    <span aria-hidden="true">•••</span><span class="sr-only">Открыть чат</span>
  </button>
  <section id="chat" hidden aria-labelledby="chat-title">
    <header>
      <h1 id="chat-title">{name}</h1>
      <button id="close-chat" type="button" aria-label="Свернуть чат">−</button>
    </header>
    <div id="messages" role="log" aria-live="polite">
      <p id="welcome">{escape(channel.welcome_text)}</p>
    </div>
    <p id="status" role="status"></p>
    <form id="composer" hidden>
      <label class="sr-only" for="message">Сообщение</label>
      <input id="message" type="text" maxlength="2000" autocomplete="off" required
             placeholder="Напишите сообщение">
      <button type="submit">Отправить</button>
    </form>
    <button type="button" id="retry" hidden>Повторить</button>
    <button type="button" id="restart" hidden>Начать новый диалог</button>
  </section>
</body>
</html>"""
    ancestors = " ".join(channel.allowed_origins)
    return HTMLResponse(
        html,
        headers={
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; "
            f"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors {ancestors}",
            "Referrer-Policy": "no-referrer",
        },
    )


@router.post("/channels/{channel_id}/sessions", status_code=201, response_model=WidgetSession)
async def start_session(
    channel_id: UUID, body: SessionInput, request: Request, key: IdempotencyKey
) -> WidgetSession:
    require_frame_origin(request)
    channel = await channel_for(request, channel_id)
    if not channel.enabled or body.parent_origin not in channel.allowed_origins:
        raise unavailable(403)
    access = WidgetAccess(
        channel.id,
        channel.organization_id,
        channel.agent_id,
        body.parent_origin,
        client_ip(request),
    )
    try:
        result = await service(request).start_public(access, key)
    except ArtifactUnavailable, ExecutionError, InvalidVersionFlow:
        raise unavailable(503) from None
    assert result.credential is not None
    claim = WidgetGrant(
        channel_id=access.id,
        organization_id=access.organization_id,
        agent_id=access.agent_id,
        conversation_id=result.conversation_id,
        origin=access.origin,
        credential=result.credential,
    )
    return WidgetSession(
        **view(request, result).model_dump(),
        session_token=issue_widget_token(request.app.state.settings, claim),
    )


@router.post("/conversations/{conversation_id}/messages", response_model=WidgetSession)
async def send_widget_message(
    conversation_id: UUID,
    body: RuntimeMessageInput,
    request: Request,
    credential: Bearer,
    key: IdempotencyKey,
) -> WidgetSession:
    require_frame_origin(request)
    if credential is None:
        raise unavailable(401)
    claim = read_widget_token(request.app.state.settings, credential.credentials, conversation_id)
    try:
        result = await service(request).send(
            claim.access(client_ip(request)),
            claim.organization_id,
            conversation_id,
            None,
            body.text,
            claim.credential,
            body.client_message_id,
            key,
        )
    except ArtifactUnavailable, ExecutionError, InvalidVersionFlow:
        raise unavailable(503) from None
    return WidgetSession(
        **view(request, result).model_dump(),
        session_token=issue_widget_token(request.app.state.settings, claim),
    )
