from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, StrictBool, StringConstraints, model_validator
from sqlalchemy import select, tuple_

from dialogentis.agents import agent_and_draft
from dialogentis.auth.security import VerifiedUser
from dialogentis.channels.widget_models import WidgetChannel, WidgetPosition
from dialogentis.channels.widget_origins import widget_origin
from dialogentis.contracts import CursorPage, InputModel, revision_etag
from dialogentis.dependencies import Database
from dialogentis.organizations import audit, organization_context

router = APIRouter(
    prefix="/api/v1/organizations/{org_id}/agents/{agent_id}/widget-channels",
    tags=["widget-channels"],
)

type ChannelName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]
type WelcomeText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]
type AccentColor = Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")]
type Origins = Annotated[
    list[Annotated[str, StringConstraints(min_length=1, max_length=512)]],
    Field(min_length=1, max_length=20),
]


class WidgetInput(InputModel):
    name: ChannelName
    allowed_origins: Origins
    welcome_text: WelcomeText = ""
    accent_color: AccentColor = "#da694e"
    position: WidgetPosition = "bottom-right"
    enabled: StrictBool = True


class WidgetPatch(InputModel):
    name: ChannelName | None = None
    allowed_origins: Origins | None = None
    welcome_text: WelcomeText | None = None
    accent_color: AccentColor | None = None
    position: WidgetPosition | None = None
    enabled: StrictBool | None = None

    @model_validator(mode="after")
    def valid_changes(self):
        if not self.model_fields_set or any(
            getattr(self, name) is None for name in self.model_fields_set
        ):
            raise ValueError("Invalid widget update")
        return self


class WidgetView(BaseModel):
    id: UUID
    agent_id: UUID
    name: str
    allowed_origins: list[str]
    welcome_text: str
    accent_color: str
    position: WidgetPosition
    enabled: bool
    revision: int
    created_at: datetime
    updated_at: datetime


def widget_query(org_id: UUID, agent_id: UUID):
    return select(WidgetChannel).where(
        WidgetChannel.organization_id == org_id, WidgetChannel.agent_id == agent_id
    )


async def find_widget(
    db: Database, org_id: UUID, agent_id: UUID, channel_id: UUID
) -> WidgetChannel:
    widget = await db.scalar(widget_query(org_id, agent_id).where(WidgetChannel.id == channel_id))
    if widget is None:
        raise HTTPException(404, "Виджет не найден")
    return widget


def view(widget: WidgetChannel, response: Response) -> WidgetView:
    response.headers["ETag"] = revision_etag(widget.revision)
    return WidgetView.model_validate(widget, from_attributes=True)


def normalize(payload: WidgetInput | WidgetPatch, request: Request) -> dict:
    data = payload.model_dump(exclude_unset=isinstance(payload, WidgetPatch))
    if "allowed_origins" in data:
        try:
            development = request.app.state.settings.app_env == "development"
            data["allowed_origins"] = list(
                dict.fromkeys(
                    widget_origin(origin, development=development)
                    for origin in data["allowed_origins"]
                )
            )
        except ValueError:
            raise HTTPException(
                422, "Укажите адреса сайтов с HTTPS, без пути, параметров и подстановок"
            ) from None
    if "accent_color" in data:
        data["accent_color"] = data["accent_color"].lower()
    return data


@router.get("", response_model=CursorPage[WidgetView])
async def list_widgets(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: UUID | None = None,
):
    await organization_context(db, user, org_id, "manage_channels")
    await agent_and_draft(db, org_id, agent_id)
    query = widget_query(org_id, agent_id)
    if cursor is not None:
        before = await find_widget(db, org_id, agent_id, cursor)
        query = query.where(
            tuple_(WidgetChannel.created_at, WidgetChannel.id)
            < tuple_(before.created_at, before.id)
        )
    rows = (
        await db.scalars(
            query.order_by(WidgetChannel.created_at.desc(), WidgetChannel.id.desc()).limit(
                limit + 1
            )
        )
    ).all()
    return CursorPage[WidgetView](
        items=[WidgetView.model_validate(row, from_attributes=True) for row in rows[:limit]],
        next_cursor=str(rows[limit - 1].id) if len(rows) > limit else None,
    )


@router.post("", response_model=WidgetView, status_code=201)
async def create_widget(
    org_id: UUID,
    agent_id: UUID,
    body: WidgetInput,
    request: Request,
    response: Response,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id, "manage_channels", lock=True)
    agent, _ = await agent_and_draft(db, org_id, agent_id)
    if agent.status != "active":
        raise HTTPException(409, "Сначала восстановите агента из архива")
    now = datetime.now(UTC)
    widget = WidgetChannel(
        organization_id=org_id,
        agent_id=agent_id,
        revision=1,
        created_at=now,
        updated_at=now,
        updated_by=user.id,
        **normalize(body, request),
    )
    db.add(widget)
    await db.flush()
    audit(db, org_id, user, "widget.created", widget.id, details={"agent_id": str(agent_id)})
    return view(widget, response)


@router.get("/{channel_id}", response_model=WidgetView)
async def get_widget(
    org_id: UUID,
    agent_id: UUID,
    channel_id: UUID,
    response: Response,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id, "manage_channels")
    return view(await find_widget(db, org_id, agent_id, channel_id), response)


@router.patch("/{channel_id}", response_model=WidgetView)
async def update_widget(
    org_id: UUID,
    agent_id: UUID,
    channel_id: UUID,
    body: WidgetPatch,
    request: Request,
    response: Response,
    user: VerifiedUser,
    db: Database,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
):
    await organization_context(db, user, org_id, "manage_channels", lock=True)
    widget = await find_widget(db, org_id, agent_id, channel_id)
    if if_match is None:
        raise HTTPException(428, "Укажите редакцию виджета")
    if if_match != revision_etag(widget.revision):
        raise HTTPException(412, "Настройки виджета изменились. Обновите страницу")
    changes = {
        name: value
        for name, value in normalize(body, request).items()
        if getattr(widget, name) != value
    }
    if changes.get("enabled"):
        agent, _ = await agent_and_draft(db, org_id, agent_id)
        if agent.status != "active":
            raise HTTPException(409, "Сначала восстановите агента из архива")
    if changes:
        for name, value in changes.items():
            setattr(widget, name, value)
        widget.revision += 1
        widget.updated_at = datetime.now(UTC)
        widget.updated_by = user.id
        audit(db, org_id, user, "widget.updated", widget.id, details={"fields": sorted(changes)})
    return view(widget, response)


@router.post("/{channel_id}/disable", status_code=204)
async def disable_widget(
    org_id: UUID, agent_id: UUID, channel_id: UUID, user: VerifiedUser, db: Database
) -> None:
    await organization_context(db, user, org_id, "manage_channels", lock=True)
    widget = await find_widget(db, org_id, agent_id, channel_id)
    if widget.enabled:
        widget.enabled = False
        widget.revision += 1
        widget.updated_at = datetime.now(UTC)
        widget.updated_by = user.id
        audit(db, org_id, user, "widget.disabled", widget.id)
