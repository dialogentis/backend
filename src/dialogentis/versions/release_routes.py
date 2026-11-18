from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, StrictBool, StringConstraints
from sqlalchemy import select, tuple_

from dialogentis.agents import agent_and_draft
from dialogentis.api_errors import error_response
from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage, InputModel, revision_etag
from dialogentis.dependencies import Database
from dialogentis.models import Agent
from dialogentis.organizations import audit, organization_context
from dialogentis.versions.release_check import ReleaseCheck, ReleasePin, check_release
from dialogentis.versions.release_models import Release, ReleaseVersion

router = APIRouter(prefix="/api/v1/organizations/{org_id}/agents/{agent_id}", tags=["releases"])
IfMatch = Annotated[str | None, Header(alias="If-Match")]
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class PublishInput(InputModel):
    version_id: UUID
    warning_acknowledgement: StrictBool = False
    reason: Reason | None = None


class RollbackInput(InputModel):
    release_id: UUID
    reason: Reason


class ReleaseView(BaseModel):
    id: UUID
    agent_id: UUID
    root_version_id: UUID
    dependency_manifest: list[ReleasePin]
    created_by: UUID
    created_at: datetime
    reason: str | None


class CurrentRelease(BaseModel):
    current_release_id: UUID


def view(release: Release) -> ReleaseView:
    return ReleaseView.model_validate(release, from_attributes=True)


async def _editable_agent(
    db: Database, user: VerifiedUser, org_id: UUID, agent_id: UUID, if_match: str | None
) -> Agent:
    await organization_context(db, user, org_id, "publish_agents", lock=True)
    agent, _ = await agent_and_draft(db, org_id, agent_id)
    if if_match is None:
        raise HTTPException(428, "Укажите редакцию агента")
    if if_match != revision_etag(agent.revision):
        raise HTTPException(412, "Агент уже изменён. Обновите страницу")
    if agent.status != "active":
        raise HTTPException(409, "Сначала восстановите агента из архива")
    return agent


@router.get("/versions/{version_id}/release-check", response_model=ReleaseCheck)
async def release_readiness(
    org_id: UUID,
    agent_id: UUID,
    version_id: UUID,
    request: Request,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id, "publish_agents")
    agent, _ = await agent_and_draft(db, org_id, agent_id)
    return await check_release(db, agent, version_id, request.app.state.settings)


@router.post("/releases", status_code=201, response_model=ReleaseView)
async def publish(
    org_id: UUID,
    agent_id: UUID,
    body: PublishInput,
    request: Request,
    response: Response,
    user: VerifiedUser,
    db: Database,
    if_match: IfMatch = None,
):
    agent = await _editable_agent(db, user, org_id, agent_id, if_match)
    check = await check_release(db, agent, body.version_id, request.app.state.settings)
    if check.errors:
        return error_response(
            request,
            409,
            "release_not_ready",
            "Версия пока не готова к публикации",
            [issue.model_dump(mode="json") for issue in check.errors[:100]],
        )
    if check.warnings and (not body.warning_acknowledgement or body.reason is None):
        return error_response(
            request,
            409,
            "release_quality_confirmation_required",
            "Подтвердите предупреждения о качестве и укажите причину публикации",
            [issue.model_dump(mode="json") for issue in check.warnings[:100]],
        )
    release = Release(
        id=uuid4(),
        organization_id=org_id,
        agent_id=agent_id,
        root_version_id=body.version_id,
        dependency_manifest=[
            pin.model_dump(mode="json")
            for pin in sorted(check.manifest, key=lambda pin: pin.version_id)
        ],
        created_by=user.id,
        reason=body.reason,
    )
    db.add(release)
    await db.flush()
    db.add_all(
        [
            ReleaseVersion(
                organization_id=org_id,
                release_agent_id=agent_id,
                release_id=release.id,
                agent_id=pin.agent_id,
                version_id=pin.version_id,
            )
            for pin in check.manifest
        ]
    )
    previous = agent.current_release_id
    agent.current_release_id = release.id
    agent.revision += 1
    audit(
        db,
        org_id,
        user,
        "agent.published",
        release.id,
        details={
            "agent_id": str(agent_id),
            "previous_release_id": str(previous) if previous else None,
            "reason": body.reason,
            "acknowledged_warnings": sorted({issue.code for issue in check.warnings}),
        },
    )
    await db.flush()
    response.headers["ETag"] = revision_etag(agent.revision)
    return view(release)


@router.get("/releases", response_model=CursorPage[ReleaseView])
async def list_releases(
    org_id: UUID,
    agent_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: UUID | None = None,
):
    await organization_context(db, user, org_id)
    await agent_and_draft(db, org_id, agent_id)
    scope = (Release.organization_id == org_id, Release.agent_id == agent_id)
    query = select(Release).where(*scope)
    if cursor is not None:
        before = await db.scalar(select(Release).where(*scope, Release.id == cursor))
        if before is None:
            raise HTTPException(404, "Публикация не найдена")
        query = query.where(
            tuple_(Release.created_at, Release.id) < tuple_(before.created_at, before.id)
        )
    rows = (
        await db.scalars(
            query.order_by(Release.created_at.desc(), Release.id.desc()).limit(limit + 1)
        )
    ).all()
    return CursorPage[ReleaseView](
        items=[view(row) for row in rows[:limit]],
        next_cursor=str(rows[limit - 1].id) if len(rows) > limit else None,
    )


@router.post("/rollback", response_model=CurrentRelease)
async def rollback(
    org_id: UUID,
    agent_id: UUID,
    body: RollbackInput,
    response: Response,
    user: VerifiedUser,
    db: Database,
    if_match: IfMatch = None,
):
    agent = await _editable_agent(db, user, org_id, agent_id, if_match)
    release = await db.scalar(
        select(Release).where(
            Release.id == body.release_id,
            Release.organization_id == org_id,
            Release.agent_id == agent_id,
        )
    )
    if release is None:
        raise HTTPException(404, "Публикация не найдена")
    if agent.current_release_id != release.id:
        archived = await db.scalar(
            select(Agent.id)
            .join(
                ReleaseVersion,
                (ReleaseVersion.agent_id == Agent.id) & (ReleaseVersion.organization_id == org_id),
            )
            .where(
                Agent.organization_id == org_id,
                Agent.status == "archived",
                ReleaseVersion.release_id == release.id,
            )
            .limit(1)
        )
        if archived is not None:
            raise HTTPException(409, "Сначала восстановите из архива зависимого агента")
        previous = agent.current_release_id
        agent.current_release_id = release.id
        agent.revision += 1
        audit(
            db,
            org_id,
            user,
            "agent.rolled_back",
            release.id,
            details={
                "agent_id": str(agent_id),
                "previous_release_id": str(previous) if previous else None,
                "reason": body.reason,
            },
        )
        await db.flush()
    response.headers["ETag"] = revision_etag(agent.revision)
    return CurrentRelease(current_release_id=release.id)
