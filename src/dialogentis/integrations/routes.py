from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from dialogentis.auth.security import VerifiedUser
from dialogentis.contracts import CursorPage, InputModel, revision_etag
from dialogentis.dependencies import Database
from dialogentis.integrations.ledger_models import ActionExecution
from dialogentis.integrations.models import Integration
from dialogentis.integrations.schema import IntegrationInput, Method
from dialogentis.integrations.secrets import IntegrationSecretError, seal_secret
from dialogentis.models import AgentDraft
from dialogentis.organizations import audit, organization_context
from dialogentis.versions.models import AgentVersion

router = APIRouter(prefix="/api/v1/organizations/{org_id}/integrations", tags=["integrations"])


class IntegrationPatch(InputModel):
    name: StrictStr | None = None
    base_url: StrictStr | None = None
    methods: list[Method] | None = None
    paths: list[StrictStr] | None = None
    response_fields: list[StrictStr] | None = None
    enabled: StrictBool | None = None

    @model_validator(mode="after")
    def valid_changes(self):
        if not self.model_fields_set or any(
            getattr(self, name) is None for name in self.model_fields_set
        ):
            raise ValueError("Invalid changes")
        return self


class SecretInput(InputModel):
    secret: SecretStr = Field(repr=False)


class IntegrationView(BaseModel):
    id: UUID
    name: str
    base_url: str
    methods: list[str]
    paths: list[str]
    response_fields: list[str]
    enabled: bool
    has_secret: bool
    revision: int
    created_at: datetime


def view(item: Integration) -> IntegrationView:
    return IntegrationView(
        id=item.id,
        name=item.name,
        base_url=item.base_url,
        methods=item.methods,
        paths=item.paths,
        response_fields=item.response_fields,
        enabled=item.enabled,
        has_secret=item.encrypted_secret is not None,
        revision=item.revision,
        created_at=item.created_at,
    )


async def find(db: Database, org_id: UUID, integration_id: UUID) -> Integration:
    item = await db.scalar(
        select(Integration).where(
            Integration.id == integration_id, Integration.organization_id == org_id
        )
    )
    if item is None:
        raise HTTPException(404, "Интеграция не найдена")
    return item


def check_revision(item: Integration, if_match: str | None) -> None:
    if if_match is None:
        raise HTTPException(428, "Укажите версию интеграции")
    if if_match != revision_etag(item.revision):
        raise HTTPException(412, "Интеграция изменилась. Обновите страницу")


@router.get("", response_model=CursorPage[IntegrationView])
async def list_integrations(
    org_id: UUID,
    user: VerifiedUser,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: UUID | None = None,
):
    await organization_context(db, user, org_id)
    query = (
        select(Integration)
        .where(Integration.organization_id == org_id)
        .order_by(Integration.id)
        .limit(limit + 1)
    )
    if cursor is not None:
        query = query.where(Integration.id > cursor)
    items = (await db.scalars(query)).all()
    return {
        "items": [view(item) for item in items[:limit]],
        "next_cursor": str(items[limit - 1].id) if len(items) > limit else None,
    }


@router.post("", response_model=IntegrationView, status_code=201)
async def create_integration(
    org_id: UUID,
    body: IntegrationInput,
    request: Request,
    response: Response,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id, "manage_integrations", lock=True)
    item = Integration(id=uuid4(), organization_id=org_id, **body.model_dump(exclude={"secret"}))
    if body.secret is not None:
        try:
            item.encrypted_secret = seal_secret(
                body.secret.get_secret_value(), org_id, item.id, request.app.state.settings
            )
        except IntegrationSecretError:
            raise HTTPException(422, "Недопустимый секрет или ключ шифрования недоступен") from None
    try:
        async with db.begin_nested():
            db.add(item)
            await db.flush()
    except IntegrityError:
        raise HTTPException(409, "Интеграция с таким названием уже существует") from None
    audit(db, org_id, user, "integration.created", item.id)
    response.headers["ETag"] = revision_etag(item.revision)
    return view(item)


@router.get("/{integration_id}", response_model=IntegrationView)
async def get_integration(
    org_id: UUID,
    integration_id: UUID,
    response: Response,
    user: VerifiedUser,
    db: Database,
):
    await organization_context(db, user, org_id)
    item = await find(db, org_id, integration_id)
    response.headers["ETag"] = revision_etag(item.revision)
    return view(item)


@router.patch("/{integration_id}", response_model=IntegrationView)
async def update_integration(
    org_id: UUID,
    integration_id: UUID,
    body: IntegrationPatch,
    response: Response,
    user: VerifiedUser,
    db: Database,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
):
    await organization_context(db, user, org_id, "manage_integrations", lock=True)
    item = await find(db, org_id, integration_id)
    check_revision(item, if_match)
    try:
        data = IntegrationInput.model_validate(
            {
                "name": item.name,
                "base_url": item.base_url,
                "methods": item.methods,
                "paths": item.paths,
                "response_fields": item.response_fields,
                "enabled": item.enabled,
            }
            | body.model_dump(exclude_unset=True)
        )
    except ValidationError:
        raise HTTPException(422, "Проверьте настройки интеграции") from None
    for name, value in data.model_dump(exclude={"secret"}).items():
        setattr(item, name, value)
    item.revision += 1
    try:
        async with db.begin_nested():
            await db.flush()
    except IntegrityError:
        raise HTTPException(409, "Интеграция с таким названием уже существует") from None
    audit(db, org_id, user, "integration.updated", item.id)
    response.headers["ETag"] = revision_etag(item.revision)
    return view(item)


@router.post("/{integration_id}/rotate-secret", response_model=IntegrationView)
async def rotate_secret(
    org_id: UUID,
    integration_id: UUID,
    body: SecretInput,
    request: Request,
    response: Response,
    user: VerifiedUser,
    db: Database,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
):
    await organization_context(db, user, org_id, "manage_integrations", lock=True)
    item = await find(db, org_id, integration_id)
    check_revision(item, if_match)
    try:
        item.encrypted_secret = seal_secret(
            body.secret.get_secret_value(), org_id, item.id, request.app.state.settings
        )
    except IntegrationSecretError:
        raise HTTPException(422, "Недопустимый секрет или ключ шифрования недоступен") from None
    item.revision += 1
    audit(db, org_id, user, "integration.secret_rotated", item.id)
    await db.flush()
    response.headers["ETag"] = revision_etag(item.revision)
    return view(item)


@router.delete("/{integration_id}", status_code=204)
async def delete_integration(
    org_id: UUID,
    integration_id: UUID,
    user: VerifiedUser,
    db: Database,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
):
    await organization_context(db, user, org_id, "manage_integrations", lock=True)
    item = await find(db, org_id, integration_id)
    check_revision(item, if_match)
    if (
        await db.scalar(
            select(ActionExecution.id)
            .where(
                ActionExecution.organization_id == org_id,
                ActionExecution.integration_id == integration_id,
            )
            .limit(1)
        )
        is not None
    ):
        raise HTTPException(409, "Интеграция используется в диалогах. Отключите её вместо удаления")
    for model, node_path in (
        (
            AgentDraft,
            "configuration #> '{flow,nodes}'",
        ),
        (
            AgentVersion,
            "snapshot #> '{config,flow,nodes}'",
        ),
    ):
        referenced = await db.scalar(
            select(model.id if model is AgentVersion else AgentDraft.agent_id)
            .where(model.organization_id == org_id)
            .where(
                text(
                    "EXISTS (SELECT 1 FROM jsonb_array_elements("
                    "CASE WHEN jsonb_typeof("
                    + node_path
                    + ") = 'array' THEN "
                    + node_path
                    + " ELSE '[]'::jsonb END) AS node "
                    "WHERE node->>'type' = 'http_action' AND "
                    "translate(replace(replace(lower(node->'config'->>'integration_id'), "
                    "'urn:', ''), 'uuid:', ''), '-{}', '') = :ref)"
                )
            )
            .params(ref=integration_id.hex)
            .limit(1)
        )
        if referenced is not None:
            raise HTTPException(
                409, "Интеграция используется в сценарии. Отключите её вместо удаления"
            )
    await db.delete(item)
    audit(db, org_id, user, "integration.deleted", integration_id)
