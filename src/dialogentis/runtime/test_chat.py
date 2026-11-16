"""Internal test-chat orchestration; not an HTTP API or an idempotency boundary."""

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import read_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings
from dialogentis.db import unit_of_work
from dialogentis.jobs.models import Job
from dialogentis.models import User
from dialogentis.nlu.model_store import ModelIdentity
from dialogentis.organizations import organization_context
from dialogentis.runtime.conversations import (
    commit_turn,
    create_conversation,
    read_conversation,
    reserve_turn,
)
from dialogentis.runtime.core import RuntimeResult, TraceEvent
from dialogentis.runtime.executor import ExecutionError, RuntimeExecutor
from dialogentis.runtime.models import TestConversation
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import version_status


@dataclass(frozen=True)
class ChatResponse:
    conversation_id: UUID
    revision: int
    status: str
    messages: tuple[str, ...] = field(repr=False)
    trace: tuple[TraceEvent, ...] = field(repr=False)
    error_code: str | None = None


@dataclass(frozen=True)
class _Bundle:
    identity: ModelIdentity
    snapshot: dict = field(repr=False)
    config_hash: str
    model: bytes = field(repr=False)


class TestChatService:
    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        executor: RuntimeExecutor,
    ):
        self.factory = factory
        self.settings = settings
        self.executor = executor

    async def _user(self, db: AsyncSession, user_id: UUID) -> User:
        user = await db.get(User, user_id)
        if user is None or not user.email_verified:
            raise HTTPException(403, "Подтвердите адрес почты")
        return user

    async def _bundle(
        self, db: AsyncSession, user: User, org_id: UUID, agent_id: UUID, version_id: UUID
    ) -> _Bundle:
        await organization_context(db, user, org_id, lock=True)
        pair = (
            await db.execute(
                select(AgentVersion, Job)
                .join(Job, Job.id == AgentVersion.job_id)
                .where(
                    AgentVersion.organization_id == org_id,
                    AgentVersion.agent_id == agent_id,
                    AgentVersion.id == version_id,
                    Job.organization_id == org_id,
                )
            )
        ).one_or_none()
        if pair is None:
            raise HTTPException(404, "Версия агента не найдена")
        version, job = pair
        if version_status(version, job) != "ready" or version.model_artifact_id is None:
            raise HTTPException(409, "Версия агента ещё не готова")
        identity = ModelIdentity(org_id, agent_id, version_id, version.model_artifact_id)
        content = await read_artifact(
            db,
            ArtifactStorage(self.settings.artifact_root),
            org_id,
            identity.artifact_id,
            max_bytes=self.settings.max_model_artifact_bytes,
            expected_kind="model",
        )
        return _Bundle(identity, version.snapshot, version.config_hash, content)

    async def _execute(self, action: str, bundle: _Bundle, state=None, text=None) -> RuntimeResult:
        if action == "start":
            return await self.executor.execute(
                action="start",
                snapshot=bundle.snapshot,
                config_hash=bundle.config_hash,
                identity=bundle.identity,
                model=bundle.model,
                settings=self.settings,
            )
        return await self.executor.execute(
            action="send",
            snapshot=bundle.snapshot,
            config_hash=bundle.config_hash,
            identity=bundle.identity,
            model=bundle.model,
            settings=self.settings,
            state=state,
            text=text,
        )

    async def start(
        self, user_id: UUID, org_id: UUID, agent_id: UUID, version_id: UUID
    ) -> ChatResponse:
        try:
            async with asyncio.timeout(25):
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    bundle = await self._bundle(
                        db, await self._user(db, user_id), org_id, agent_id, version_id
                    )
                result = await self._execute("start", bundle)
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    saved = await create_conversation(
                        db,
                        await self._user(db, user_id),
                        org_id,
                        agent_id,
                        version_id,
                        result.state,
                        self.settings,
                    )
                return ChatResponse(
                    saved.id,
                    saved.revision,
                    result.state.status,
                    result.messages,
                    result.trace,
                    result.error_code,
                )
        except TimeoutError:
            raise ExecutionError("runtime_request_timeout") from None

    async def send(
        self, user_id: UUID, org_id: UUID, conversation_id: UUID, revision: int, text: str
    ) -> ChatResponse:
        if not isinstance(text, str) or not 1 <= len(text) <= 2000:
            raise ExecutionError("runtime_invalid_request")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise ExecutionError("runtime_invalid_request") from None
        try:
            async with asyncio.timeout(25):
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    user = await self._user(db, user_id)
                    saved = await read_conversation(
                        db, user, org_id, conversation_id, self.settings
                    )
                    if saved.revision != revision:
                        raise HTTPException(409, "Диалог изменился. Обновите состояние")
                    row = await db.get(TestConversation, conversation_id)
                    assert row is not None
                    bundle = await self._bundle(db, user, org_id, row.agent_id, row.version_id)
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    reservation = await reserve_turn(
                        db,
                        await self._user(db, user_id),
                        org_id,
                        conversation_id,
                        revision,
                        self.settings,
                    )
                result = await self._execute("send", bundle, reservation.state, text)
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    saved = await commit_turn(
                        db,
                        await self._user(db, user_id),
                        org_id,
                        conversation_id,
                        revision,
                        reservation.token,
                        result.state,
                        self.settings,
                    )
                return ChatResponse(
                    saved.id,
                    saved.revision,
                    result.state.status,
                    result.messages,
                    result.trace,
                    result.error_code,
                )
        except TimeoutError:
            raise ExecutionError("runtime_request_timeout") from None
