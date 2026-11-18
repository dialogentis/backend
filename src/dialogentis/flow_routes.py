import asyncio
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt

from dialogentis.agents import agent_and_draft
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import InputModel
from dialogentis.datasets.routes import ValidationIssue, issue_view
from dialogentis.dependencies import Database
from dialogentis.flow_validation import validate_flow
from dialogentis.organizations import organization_context
from dialogentis.versions.handoff_dependencies import validate_handoff_dependencies

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["flow"])


class FlowValidationInput(InputModel):
    revision: StrictInt = Field(ge=1)


class FlowValidationView(BaseModel):
    revision: int
    scope: Literal["flow_structure"] = "flow_structure"
    complete: Literal[False] = False
    truncated: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]


@router.post("/draft/flow/validate", response_model=FlowValidationView)
async def validate_draft_flow(
    org_id: UUID, agent_id: UUID, body: FlowValidationInput, user: VerifiedUser, db: Database
):
    await organization_context(db, user, org_id)
    _, draft = await agent_and_draft(db, org_id, agent_id)
    if draft.revision != body.revision:
        raise HTTPException(409, "Черновик изменился. Обновите страницу")
    errors, warnings = await asyncio.to_thread(validate_flow, draft.configuration)
    if not errors:
        errors.extend(
            await validate_handoff_dependencies(db, org_id, agent_id, draft.configuration)
        )
    limit = 1000
    return FlowValidationView(
        revision=body.revision,
        truncated=len(errors) + len(warnings) > limit,
        errors=[issue_view(issue) for issue in errors[:limit]],
        warnings=[issue_view(issue) for issue in warnings[: max(0, limit - len(errors))]],
    )
