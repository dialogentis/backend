"""Execute a validated HTTP block outside the runtime child process."""

import asyncio
import json
from dataclasses import dataclass, field
from time import monotonic
from typing import cast
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.http_action import HTTP_OUTPUTS, HttpActionConfig, map_http_response
from dialogentis.integrations.http_client import IntegrationRequestError, request_json
from dialogentis.integrations.ledger import Outcome, begin_action, finish_action
from dialogentis.integrations.models import Integration
from dialogentis.integrations.schema import IntegrationInput, path_allowed
from dialogentis.integrations.secrets import IntegrationSecretError, open_secret
from dialogentis.models import User
from dialogentis.organizations import organization_context
from dialogentis.runtime.conversations import PublicAccess
from dialogentis.runtime.idempotency import ReservedMessage
from dialogentis.slots import Slot


@dataclass
class LiveHttpTurn:
    org_id: UUID
    user_id: UUID | PublicAccess
    credential: str = field(repr=False)
    reserved: ReservedMessage = field(repr=False)
    deadline: float
    ordinal: int = 0
    external_http_seconds: float = 0

    async def _principal(self, db: AsyncSession):
        if isinstance(self.user_id, PublicAccess):
            await self.user_id.lock_organization(db, self.org_id)
            return self.user_id
        user = await db.get(User, self.user_id)
        if user is None or not user.email_verified:
            raise HTTPException(403, "Подтвердите адрес почты")
        await organization_context(db, user, self.org_id, "manage_integrations", lock=True)
        return user

    async def execute(
        self,
        factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        node_id: UUID,
        config: HttpActionConfig,
        slots: dict[UUID, Slot],
        inputs: dict,
    ) -> tuple[str, dict]:
        query = {key: inputs[slots[ref].code] for key, ref in config.query.items()}
        body = {key: inputs[slots[ref].code] for key, ref in config.body.items()}
        if None in query.values() or None in body.values():
            return "http_error", {}
        query = {
            key: value if isinstance(value, str) else json.dumps(value)
            for key, value in query.items()
        }
        actor = self.user_id if isinstance(self.user_id, UUID) else None
        async with unit_of_work(factory, organization_id=self.org_id, user_id=actor) as db:
            user = await self._principal(db)
            item = await db.scalar(
                select(Integration).where(
                    Integration.organization_id == self.org_id,
                    Integration.id == config.integration_id,
                )
            )
            if item is None or not item.enabled or config.method not in item.methods:
                return "http_error", {}
            try:
                policy = IntegrationInput.model_validate(item, from_attributes=True)
                if not path_allowed(config.path, policy.paths):
                    return "http_error", {}
                secret = (
                    open_secret(item.encrypted_secret, self.org_id, item.id, settings)
                    if item.encrypted_secret is not None
                    else None
                )
            except ValidationError, ValueError, IntegrationSecretError:
                return "http_error", {}
            if set(config.response_mapping) - set(policy.response_fields):
                return "invalid_response", {}
            if self.deadline <= asyncio.get_running_loop().time():
                return "timeout", {}
            action = await begin_action(
                db,
                user,
                self.org_id,
                self.reserved.reservation.id,
                self.credential,
                self.reserved,
                item.id,
                node_id,
                self.ordinal + 1,
                config.method,
            )
        self.ordinal += 1

        updates = {}
        status = None
        remaining = self.deadline - asyncio.get_running_loop().time()
        started = monotonic()
        try:
            if remaining <= 0:
                raise IntegrationRequestError("timeout")
            response = await request_json(
                policy,
                config.method,
                config.path,
                query,
                body or None,
                secret=secret,
                execution_id=action.id,
                supports_idempotency=policy.supports_idempotency,
                remaining_seconds=min(20, remaining),
            )
            mapped = map_http_response(config, policy.response_fields, slots, response.value)
            if mapped is None:
                raise IntegrationRequestError("invalid_response")
            updates = mapped
            outcome: Outcome = "succeeded"
            status = response.status
        except IntegrationRequestError as exc:
            outcome = cast(Outcome, exc.code)
        finally:
            self.external_http_seconds += monotonic() - started

        async with unit_of_work(factory, organization_id=self.org_id, user_id=actor) as db:
            user = await self._principal(db)
            await finish_action(
                db, user, self.org_id, self.credential, self.reserved, action, outcome, status
            )
        if outcome == "succeeded":
            return "success", updates
        return (outcome if outcome in HTTP_OUTPUTS else "http_error"), {}
