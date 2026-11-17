from copy import deepcopy
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, ValidationError, field_validator
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


class AgentCopy(InputModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank")
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
                    "config": {
                        "resume_node_id": switch,
                        "text": "Не удалось распознать запрос. Попробуйте переформулировать.",
                    },
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


def copy_config(config: dict) -> dict:
    """Copy only configurations whose internal references are understood."""
    from dialogentis.ask_slot import AskSlotDraftConfig
    from dialogentis.conditions import (
        ConditionDraftConfig,
        remap_condition_expression,
        validate_draft_condition,
    )
    from dialogentis.datasets.schema import Dataset
    from dialogentis.drafts import DraftConfig
    from dialogentis.http_action import HTTP_OUTPUTS, HttpActionDraftConfig, check_http_references
    from dialogentis.set_slot import SetSlotDraftConfig
    from dialogentis.slots import validate_slots

    try:
        result = DraftConfig.model_validate(config).model_dump(mode="json")
        dataset = Dataset.model_validate(
            {key: result[key] for key in ("intents", "entities", "examples")}
        )
    except ValidationError as exc:
        raise HTTPException(409, "Нельзя скопировать некорректный черновик") from exc
    flow = result["flow"]
    slots, slot_errors = validate_slots(result["slots"])
    if slot_errors:
        raise HTTPException(409, "Нельзя скопировать некорректные параметры")
    allowed = {
        "start": set(),
        "switch_intent": set(),
        "fallback": {"resume_node_id", "text", "example"},
        "say": {"text"},
        "end": {"text"},
        "ask_slot": {"slot_id", "prompt", "entity_id", "max_attempts", "checks"},
        "set_slot": {"slot_id", "operation", "source", "source_slot_id", "value"},
        "condition": {"branches"},
        "http_action": set(HttpActionDraftConfig.model_fields),
    }
    if flow["layout"]:
        raise HTTPException(409, "Копирование этого черновика пока недоступно")
    if any(
        node["type"] not in allowed or set(node["config"]) - allowed[node["type"]]
        for node in flow["nodes"]
    ):
        raise HTTPException(409, "Копирование этого черновика пока недоступно")
    node_ids = [node["id"] for node in flow["nodes"]]
    edge_ids = [edge["id"] for edge in flow["edges"]]
    if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
        raise HTTPException(409, "Нельзя скопировать сценарий с повторяющимися идентификаторами")
    mapping = {identifier: str(uuid4()) for identifier in node_ids}
    if any(
        edge["source"] not in mapping or edge["target"] not in mapping for edge in flow["edges"]
    ):
        raise HTTPException(409, "Нельзя скопировать сценарий с отсутствующими блоками")
    node_types = {node["id"]: node["type"] for node in flow["nodes"]}
    intent_ids = {str(intent.id) for intent in dataset.intents}
    outputs = {
        "start": {"next"},
        "say": {"next"},
        "switch_intent": {"fallback", *intent_ids},
        "ask_slot": {"success", "invalid", "exhausted"},
        "set_slot": {"next", "error"},
        "http_action": set(HTTP_OUTPUTS),
    }
    conditions = {}
    for node in flow["nodes"]:
        if node["type"] == "condition":
            try:
                condition = ConditionDraftConfig.model_validate(node["config"])
                validate_draft_condition(condition, {slot.id: slot for slot in slots})
            except ValidationError, ValueError:
                raise HTTPException(409, "Нельзя скопировать некорректное условие") from None
            conditions[node["id"]] = condition
    branch_mapping = {
        node_id: {str(branch.id): str(uuid4()) for branch in condition.branches}
        for node_id, condition in conditions.items()
    }
    handles = [(edge["source"], edge["source_handle"]) for edge in flow["edges"]]
    if len(handles) != len(set(handles)) or any(
        edge["source_handle"]
        not in (
            {"else", *branch_mapping[edge["source"]]}
            if node_types[edge["source"]] == "condition"
            else outputs.get(node_types[edge["source"]], set())
        )
        for edge in flow["edges"]
    ):
        raise HTTPException(409, "Копирование этого сценария пока недоступно")
    for node in flow["nodes"]:
        if node["type"] == "http_action":
            try:
                action = HttpActionDraftConfig.model_validate(node["config"])
                check_http_references(action, {slot.id: slot for slot in slots})
            except ValidationError, ValueError:
                raise HTTPException(409, "Нельзя скопировать некорректный HTTP-запрос") from None
        if "text" in node["config"] and not isinstance(node["config"]["text"], str):
            raise HTTPException(409, "Нельзя скопировать некорректный текст блока")
        if "example" in node["config"] and (
            not isinstance(node["config"]["example"], str) or len(node["config"]["example"]) > 500
        ):
            raise HTTPException(409, "Нельзя скопировать некорректный пример запроса")
        if node["type"] == "set_slot":
            try:
                assignment = SetSlotDraftConfig.model_validate(node["config"])
            except ValidationError:
                raise HTTPException(409, "Нельзя скопировать некорректный блок параметра") from None
            if any(
                ref is not None and ref not in {slot.id for slot in slots}
                for ref in (assignment.slot_id, assignment.source_slot_id)
            ):
                raise HTTPException(409, "Нельзя скопировать сценарий с отсутствующим параметром")
        if node["type"] == "ask_slot":
            try:
                ask = AskSlotDraftConfig.model_validate(node["config"])
            except ValidationError:
                raise HTTPException(409, "Нельзя скопировать некорректный блок параметра") from None
            if ask.slot_id is not None and ask.slot_id not in {slot.id for slot in slots}:
                raise HTTPException(409, "Нельзя скопировать сценарий с отсутствующим параметром")
            if ask.entity_id is not None and ask.entity_id not in {
                entity.id for entity in dataset.entities
            }:
                raise HTTPException(409, "Нельзя скопировать сценарий с отсутствующей сущностью")
        reference = node["config"].get("resume_node_id")
        if "resume_node_id" in node["config"] and (
            not isinstance(reference, str) or reference not in mapping
        ):
            raise HTTPException(409, "Нельзя скопировать сценарий с отсутствующими блоками")
    identifiers = {
        "intents": [item.id for item in dataset.intents],
        "entities": [item.id for item in dataset.entities],
        "examples": [item.id for item in dataset.examples],
        "slots": [item.id for item in slots],
    }
    if any(len(ids) != len(set(ids)) for ids in identifiers.values()):
        raise HTTPException(409, "Нельзя скопировать данные с повторяющимися идентификаторами")
    data_mapping = {
        key: {identifier: str(uuid4()) for identifier in ids} for key, ids in identifiers.items()
    }
    if any(
        (example.intent_id is not None and example.intent_id not in data_mapping["intents"])
        or any(span.entity_id not in data_mapping["entities"] for span in example.spans)
        for example in dataset.examples
    ):
        raise HTTPException(409, "Нельзя скопировать данные с отсутствующими ссылками")
    clone = deepcopy(result)
    for key in ("intents", "entities", "examples"):
        for item in clone[key]:
            item["id"] = data_mapping[key][UUID(item["id"])]
            if key == "examples":
                if item.get("intent_id") is not None:
                    item["intent_id"] = data_mapping["intents"][UUID(item["intent_id"])]
                for span in item.get("spans", []):
                    span["entity_id"] = data_mapping["entities"][UUID(span["entity_id"])]
    for item in clone["slots"]:
        item["id"] = data_mapping["slots"][UUID(item["id"])]
    clone["flow"]["flow_id"] = str(uuid4())
    for node in clone["flow"]["nodes"]:
        old_node_id = node["id"]
        if node["type"] == "http_action":
            for key in ("query", "body", "response_mapping"):
                for name, reference in node["config"].get(key, {}).items():
                    node["config"][key][name] = data_mapping["slots"][UUID(reference)]
        if node["type"] == "condition":
            for branch in node["config"]["branches"]:
                branch["id"] = branch_mapping[old_node_id][str(UUID(branch["id"]))]
                remap_condition_expression(branch["expression"], data_mapping["slots"])
        node["id"] = mapping[node["id"]]
        if "resume_node_id" in node["config"]:
            node["config"]["resume_node_id"] = mapping[node["config"]["resume_node_id"]]
        if node["type"] == "set_slot":
            for key in ("slot_id", "source_slot_id"):
                reference = node["config"].get(key)
                if reference is not None:
                    node["config"][key] = data_mapping["slots"][UUID(reference)]
        if node["type"] == "ask_slot":
            for key, collection in (("slot_id", "slots"), ("entity_id", "entities")):
                reference = node["config"].get(key)
                if reference is not None:
                    node["config"][key] = data_mapping[collection][UUID(reference)]
    for edge in clone["flow"]["edges"]:
        edge["id"] = str(uuid4())
        if node_types[edge["source"]] == "condition" and edge["source_handle"] != "else":
            edge["source_handle"] = branch_mapping[edge["source"]][edge["source_handle"]]
        if node_types[edge["source"]] == "switch_intent" and edge["source_handle"] != "fallback":
            edge["source_handle"] = data_mapping["intents"][UUID(edge["source_handle"])]
        edge["source"] = mapping[edge["source"]]
        edge["target"] = mapping[edge["target"]]
    return clone


@router.post("/{agent_id}/copy", status_code=201, response_model=AgentView)
async def copy_agent(
    org_id: UUID,
    agent_id: UUID,
    body: AgentCopy,
    user: VerifiedUser,
    db: Database,
    response: Response,
):
    await organization_context(db, user, org_id, "manage_agents", lock=True)
    original, original_draft = await agent_and_draft(db, org_id, agent_id)
    configuration = copy_config(original_draft.configuration)
    clone = Agent(
        organization_id=org_id, name=body.name, slug=body.slug, description=original.description
    )
    try:
        async with db.begin_nested():
            db.add(clone)
            await db.flush()
    except IntegrityError as exc:
        raise HTTPException(409, "Адрес агента уже занят") from exc
    draft = AgentDraft(organization_id=org_id, agent_id=clone.id, configuration=configuration)
    db.add(draft)
    await db.flush()
    audit(db, org_id, user, "agent.copied", clone.id)
    response.headers["ETag"] = revision_etag(clone.revision)
    return view(clone, draft)


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
