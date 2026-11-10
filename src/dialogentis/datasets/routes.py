import asyncio
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from dialogentis.agents import agent_and_draft
from dialogentis.api_errors import DomainValidationError
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.datasets.preview import Preview, run_preview
from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.datasets.validation import Issue
from dialogentis.dependencies import Database
from dialogentis.flow_schema import Flow
from dialogentis.flow_validation import validate_intent_outputs
from dialogentis.organizations import organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["datasets"])
DATASET_FIELDS = frozenset(
    {
        "intents",
        "entities",
        "examples",
        "id",
        "code",
        "name",
        "description",
        "enabled",
        "value_type",
        "method",
        "dictionary",
        "value",
        "synonyms",
        "pattern",
        "group",
        "text",
        "intent_id",
        "expected_fallback",
        "split",
        "spans",
        "entity_id",
        "start",
        "end",
    }
)


class ValidationInput(InputModel):
    revision: int = Field(ge=1)


class ValidationIssue(BaseModel):
    path: list[str | int]
    code: str
    message: str
    related_index: int | None = None


class ValidationView(BaseModel):
    revision: int
    scope: Literal["dataset"] = "dataset"
    complete: Literal[False] = False
    truncated: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]


class DatasetView(BaseModel):
    revision: int
    dataset: Dataset


class PreviewInput(InputModel):
    revision: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=2000)


def parse_dataset(config: dict) -> Dataset:
    return Dataset.model_validate(
        {key: config.get(key) for key in ("intents", "entities", "examples")}
    )


def safe_path(location: tuple) -> tuple[str | int, ...]:
    return tuple(
        part if isinstance(part, int) or part in DATASET_FIELDS else "<unknown>"
        for part in location
    )


def validate_configuration(config: dict) -> tuple[list[Issue], list[Issue]]:
    try:
        dataset = parse_dataset(config)
    except ValidationError as exc:
        return [
            Issue(safe_path(error["loc"]), error["type"], "Некорректное значение")
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ], []
    errors, warnings = validate_dataset(dataset)
    try:
        flow = Flow.model_validate(config.get("flow"))
    except ValidationError:
        return errors, warnings
    branch_errors, branch_warnings = validate_intent_outputs(flow, dataset.intents)
    return [*errors, *branch_errors], [*warnings, *branch_warnings]


def issue_view(issue: Issue) -> ValidationIssue:
    return ValidationIssue(
        path=list(issue.path),
        code=issue.code,
        message=issue.message,
        related_index=issue.related_index,
    )


@router.get("/draft/dataset", response_model=DatasetView)
async def get_dataset(org_id: UUID, agent_id: UUID, user: VerifiedUser, db: Database):
    await organization_context(db, user, org_id)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    try:
        dataset = parse_dataset(draft.configuration)
    except ValidationError as exc:
        errors = [
            {
                "path": ["config", *safe_path(error["loc"])],
                "code": error["type"],
                "message": "Некорректное значение",
            }
            for error in exc.errors(include_input=False, include_context=False, include_url=False)[
                :1000
            ]
        ]
        raise DomainValidationError(errors) from exc
    return DatasetView(revision=draft.revision, dataset=dataset)


@router.post("/draft/entities/{entity_id}/preview", response_model=Preview)
async def preview_draft_entity(
    org_id: UUID,
    agent_id: UUID,
    entity_id: UUID,
    body: PreviewInput,
    request: Request,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id, "manage_agents")
    _, draft = await agent_and_draft(db, org_id, agent_id)
    if draft.revision != body.revision:
        raise HTTPException(409, "Черновик изменился. Обновите страницу")
    try:
        dataset = parse_dataset(draft.configuration)
    except ValidationError as exc:
        raise HTTPException(422, "Проверьте данные агента") from exc
    entity = next((item for item in dataset.entities if item.id == entity_id), None)
    if entity is None:
        raise HTTPException(404, "Сущность не найдена")
    if entity.method == "learned":
        raise HTTPException(422, "Для проверки этой сущности требуется обученная версия")
    try:
        return await run_preview(entity, body.text, request.app.state.entity_preview_slots)
    except TimeoutError as exc:
        raise HTTPException(503, "Проверка заняла слишком много времени") from exc
    except ValueError as exc:
        raise HTTPException(422, "Проверьте настройки сущности") from exc


@router.post("/draft/validate", response_model=ValidationView)
async def validate_draft(
    org_id: UUID, agent_id: UUID, body: ValidationInput, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id, "manage_agents")
    _, draft = await agent_and_draft(db, org_id, agent_id)
    if draft.revision != body.revision:
        raise HTTPException(409, "Черновик изменился. Обновите страницу")
    errors, warnings = await asyncio.to_thread(validate_configuration, draft.configuration)
    limit = 1000
    return ValidationView(
        revision=body.revision,
        truncated=len(errors) + len(warnings) > limit,
        errors=[issue_view(issue) for issue in errors[:limit]],
        warnings=[issue_view(issue) for issue in warnings[: max(0, limit - len(errors))]],
    )
