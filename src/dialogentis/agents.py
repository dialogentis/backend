from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage, InputModel, revision_etag
from dialogentis.dependencies import Database
from dialogentis.models import Agent, AgentDraft
from dialogentis.organizations import audit, organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents", tags=["agents"])


class AgentInput(InputModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank")
        return value


class AgentPatch(InputModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("Name must not be blank")
        return value.strip()

    @field_validator("description")
    @classmethod
    def nonnull_description(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("Description must not be null")
        return value


class AgentView(BaseModel):
    id: UUID
    organization_id: UUID
    name: str
    slug: str
    description: str
    language: Literal["ru"]
    status: Literal["active", "archived"]
    revision: int
    draft_revision: int


def view(agent: Agent, draft: AgentDraft) -> AgentView:
    return AgentView(
        id=agent.id,
        organization_id=agent.organization_id,
        name=agent.name,
        slug=agent.slug,
        description=agent.description,
        language="ru",
        status=agent.status,
        revision=agent.revision,
        draft_revision=draft.revision,
    )


def empty_config() -> dict:
    start, switch, fallback, end = (str(uuid4()) for _ in range(4))
    return {
        "intents": [],
        "entities": [],
        "examples": [],
        "slots": [],
        "flow": {
            "schema_version": 1,
            "flow_id": str(uuid4()),
            "nodes": [
                {"id": start, "type": "start", "config": {}, "position": {"x": 0, "y": 0}},
                {
                    "id": switch,
                    "type": "switch_intent",
                    "config": {},
                    "position": {"x": 250, "y": 0},
                },
                {
                    "id": fallback,
                    "type": "fallback",
                    "config": {"resume_node_id": switch},
                    "position": {"x": 500, "y": 0},
                },
                {"id": end, "type": "end", "config": {}, "position": {"x": 500, "y": 200}},
            ],
            "edges": [
                {"id": str(uuid4()), "source": start, "source_handle": "next", "target": switch},
                {
                    "id": str(uuid4()),
                    "source": switch,
                    "source_handle": "fallback",
                    "target": fallback,
                },
            ],
            "layout": {},
        },
    }


async def agent_and_draft(db: Database, org_id: UUID, agent_id: UUID) -> tuple[Agent, AgentDraft]:
    row = (
        await db.execute(
            select(Agent, AgentDraft)
            .join(AgentDraft, AgentDraft.agent_id == Agent.id)
            .where(Agent.id == agent_id, Agent.organization_id == org_id)
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(404, "Агент не найден")
    return row[0], row[1]


@router.get("", response_model=CursorPage[AgentView])
async def list_agents(
    org_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: UUID | None = None,
    status: Literal["active", "archived"] | None = None,
    q: Annotated[str | None, Query(max_length=120)] = None,
):
    await organization_context(db, user, org_id)
    query = (
        select(Agent, AgentDraft)
        .join(AgentDraft, AgentDraft.agent_id == Agent.id)
        .where(Agent.organization_id == org_id)
        .order_by(Agent.id)
        .limit(limit + 1)
    )
    if cursor is not None:
        query = query.where(Agent.id > cursor)
    if status is not None:
        query = query.where(Agent.status == status)
    if q and q.strip():
        search = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(
            or_(
                Agent.name.ilike(f"%{search}%", escape="\\"),
                Agent.slug.ilike(f"%{search}%", escape="\\"),
            )
        )
    rows = (await db.execute(query)).all()
    return {
        "items": [view(agent, draft) for agent, draft in rows[:limit]],
        "next_cursor": str(rows[limit - 1][0].id) if len(rows) > limit else None,
    }


@router.post("", status_code=201, response_model=AgentView)
async def create_agent(
    org_id: UUID, body: AgentInput, user: VerifiedUser, db: Database, response: Response
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    agent = Agent(
        organization_id=org_id, name=body.name, slug=body.slug, description=body.description
    )
    try:
        async with db.begin_nested():
            db.add(agent)
            await db.flush()
    except IntegrityError as exc:
        raise HTTPException(409, "Адрес агента уже занят") from exc
    draft = AgentDraft(organization_id=org_id, agent_id=agent.id, configuration=empty_config())
    db.add(draft)
    await db.flush()
    audit(db, org_id, user, "agent.created", agent.id)
    response.headers["ETag"] = revision_etag(agent.revision)
    return view(agent, draft)


@router.get("/{agent_id}", response_model=AgentView)
async def get_agent(
    org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database, response: Response
):
    await organization_context(db, user, org_id)
    agent, draft = await agent_and_draft(db, org_id, agent_id)
    response.headers["ETag"] = revision_etag(agent.revision)
    return view(agent, draft)


@router.patch("/{agent_id}", response_model=AgentView)
async def update_agent(
    org_id: UUID,
    agent_id: UUID,
    body: AgentPatch,
    user: VerifiedUser,
    db: Database,
    response: Response,
    if_match: Annotated[str, Header()],
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    agent, draft = await agent_and_draft(db, org_id, agent_id)
    if if_match != revision_etag(agent.revision):
        raise HTTPException(412, "Агент уже изменён. Обновите страницу")
    if not body.model_fields_set:
        raise HTTPException(422, "Укажите изменения")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(agent, key, value)
    agent.revision += 1
    await db.flush()
    response.headers["ETag"] = revision_etag(agent.revision)
    return view(agent, draft)


async def set_archived(
    org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database, archived: bool
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    agent, draft = await agent_and_draft(db, org_id, agent_id)
    status: Literal["active", "archived"] = "archived" if archived else "active"
    if agent.status != status:
        agent.status = status
        agent.revision += 1
        audit(db, org_id, user, "agent.archived" if archived else "agent.restored", agent.id)
        await db.flush()
    return view(agent, draft)


@router.post("/{agent_id}/archive", response_model=AgentView)
async def archive_agent(org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database):
    return await set_archived(org_id, agent_id, user, db, True)


@router.post("/{agent_id}/restore", response_model=AgentView)
async def restore_agent(org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database):
    return await set_archived(org_id, agent_id, user, db, False)
