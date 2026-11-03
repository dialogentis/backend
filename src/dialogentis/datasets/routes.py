import asyncio
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from dialogentis.agents import agent_and_draft
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.datasets.validation import Issue
from dialogentis.dependencies import Database
from dialogentis.organizations import organization_context

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["datasets"])


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


def validate_configuration(config: dict) -> tuple[list[Issue], list[Issue]]:
    try:
        dataset = Dataset.model_validate(
            {key: config.get(key) for key in ("intents", "entities", "examples")}
        )
    except ValidationError as exc:
        return [
            Issue(tuple(error["loc"]), error["type"], "Некорректное значение")
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ], []
    return validate_dataset(dataset)


def issue_view(issue: Issue) -> ValidationIssue:
    return ValidationIssue(
        path=list(issue.path),
        code=issue.code,
        message=issue.message,
        related_index=issue.related_index,
    )


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
