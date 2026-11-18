"""Internal orchestration of idempotent test-chat turns."""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import read_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings
from dialogentis.datasets.validation import Issue
from dialogentis.db import unit_of_work
from dialogentis.flow_validation import validate_flow
from dialogentis.http_action import HttpActionConfig, MockIntegration, mock_http_action
from dialogentis.integrations.models import Integration
from dialogentis.jobs.models import Job
from dialogentis.models import User
from dialogentis.nlu.model_store import ModelIdentity
from dialogentis.organizations import organization_context
from dialogentis.runtime.core import Prediction, RuntimeEngine, RuntimeResult, TraceEvent
from dialogentis.runtime.executor import ExecutionError, HttpHandler, RuntimeExecutor, RuntimeTarget
from dialogentis.runtime.idempotency import (
    Replay,
    ReplayedStart,
    complete_initial_turn,
    complete_message,
    reserve_initial_turn,
    reserve_message,
)
from dialogentis.runtime.limits import run_limited
from dialogentis.runtime.live_http import LiveHttpTurn
from dialogentis.runtime.models import TestConversation
from dialogentis.slots import validate_slots
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
    credential: str | None = field(default=None, repr=False)
    turn_id: UUID | None = None
    replayed: bool = False
    replay_available: bool = False


@dataclass(frozen=True)
class _Bundle:
    identity: ModelIdentity
    snapshot: dict = field(repr=False)
    config_hash: str
    model: bytes = field(repr=False)
    mock_integrations: list[MockIntegration] = field(repr=False)


class InvalidVersionFlow(ValueError):
    def __init__(self, errors: list[Issue]):
        super().__init__("Invalid version flow")
        self.errors = errors


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
        self._pending: set[asyncio.Task[ChatResponse]] = set()
        self._closing = False

    async def shutdown(self, grace_period: float = 30) -> None:
        self._closing = True
        if not self._pending:
            return
        _, pending = await asyncio.wait(self._pending, timeout=grace_period)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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
        errors, _ = await asyncio.to_thread(validate_flow, version.snapshot["config"])
        if errors:
            raise InvalidVersionFlow(errors)
        identity = ModelIdentity(org_id, agent_id, version_id, version.model_artifact_id)
        content = await read_artifact(
            db,
            ArtifactStorage(self.settings.artifact_root),
            org_id,
            identity.artifact_id,
            max_bytes=self.settings.max_model_artifact_bytes,
            expected_kind="model",
        )
        integration_ids = {
            HttpActionConfig.model_validate(node["config"]).integration_id
            for node in version.snapshot["config"]["flow"]["nodes"]
            if node["type"] == "http_action"
        }
        records = (
            await db.execute(
                select(
                    Integration.id,
                    Integration.enabled,
                    Integration.methods,
                    Integration.paths,
                    Integration.response_fields,
                ).where(Integration.organization_id == org_id, Integration.id.in_(integration_ids))
            )
        ).all()
        integrations = [MockIntegration.model_validate(row._mapping) for row in records]
        return _Bundle(identity, version.snapshot, version.config_hash, content, integrations)

    async def _execute(
        self,
        action: str,
        user_id: UUID,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID,
        state=None,
        text=None,
        live_http: LiveHttpTurn | None = None,
    ) -> RuntimeResult:
        async def work() -> RuntimeResult:
            async with unit_of_work(self.factory, organization_id=org_id, user_id=user_id) as db:
                bundle = await self._bundle(
                    db, await self._user(db, user_id), org_id, agent_id, version_id
                )
            return await self._execute_in_slot(action, bundle, user_id, state, text, live_http)

        try:
            async with asyncio.timeout(20):
                return await run_limited(self.factory, work)
        except TimeoutError:
            raise ExecutionError("runtime_timeout") from None

    def _http_handler(self, bundle: _Bundle, live_http: LiveHttpTurn | None) -> HttpHandler | None:
        config = bundle.snapshot["config"]
        parsed_slots, errors = validate_slots(config.get("slots", []))
        if errors:
            raise ExecutionError("runtime_invalid_request")
        slots = {slot.id: slot for slot in parsed_slots}
        actions = {
            UUID(node["id"]): HttpActionConfig.model_validate(node["config"])
            for node in config["flow"]["nodes"]
            if node["type"] == "http_action"
        }
        integrations = {item.id: item for item in bundle.mock_integrations}
        if not actions:
            return None

        async def handle_http(node_id: UUID, inputs: dict) -> tuple[str, dict]:
            http_action = actions[node_id]
            if live_http is not None:
                return await live_http.execute(
                    self.factory, self.settings, node_id, http_action, slots, inputs
                )
            values = dict(inputs)
            outcome = mock_http_action(
                http_action, integrations.get(http_action.integration_id), slots, values
            )
            updates = (
                {
                    slots[slot_id].code: values[slots[slot_id].code]
                    for slot_id in http_action.response_mapping.values()
                }
                if outcome == "success"
                else {}
            )
            return outcome, updates

        return handle_http

    async def _execute_in_slot(
        self,
        action: str,
        bundle: _Bundle,
        user_id: UUID,
        state=None,
        text=None,
        live_http: LiveHttpTurn | None = None,
    ) -> RuntimeResult:
        async def load_target(agent_id: UUID, version_id: UUID) -> RuntimeTarget | None:
            org_id = bundle.identity.organization_id
            try:
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    target = await self._bundle(
                        db, await self._user(db, user_id), org_id, agent_id, version_id
                    )
            except InvalidVersionFlow:
                return None
            except HTTPException as exc:
                if exc.status_code in (404, 409):
                    return None
                raise
            return RuntimeTarget(
                target.identity,
                target.snapshot,
                target.config_hash,
                target.model,
                self._http_handler(target, live_http),
            )

        handler = self._http_handler(bundle, live_http)
        has_handoff = any(
            node["type"] == "handoff" for node in bundle.snapshot["config"]["flow"]["nodes"]
        )
        loader = load_target if has_handoff else None
        if action == "start":
            return await self.executor.execute(
                action="start",
                snapshot=bundle.snapshot,
                config_hash=bundle.config_hash,
                identity=bundle.identity,
                model=bundle.model,
                settings=self.settings,
                http_handler=handler,
                target_loader=loader,
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
            http_handler=handler,
            target_loader=loader,
        )

    async def start(
        self,
        user_id: UUID,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID,
        *,
        idempotency_key: str | None = None,
        http_mode: str = "mock",
    ) -> ChatResponse:
        if http_mode not in ("mock", "live") or (http_mode == "live" and idempotency_key is None):
            raise HTTPException(422, "Для реальных вызовов укажите ключ запуска")
        return await self._run_detached(
            self._start(user_id, org_id, agent_id, version_id, idempotency_key, http_mode)
        )

    async def _start(
        self,
        user_id: UUID,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID,
        idempotency_key: str | None,
        http_mode: str,
    ) -> ChatResponse:
        deadline = asyncio.get_running_loop().time() + 20

        async def work() -> ChatResponse:
            async with unit_of_work(self.factory, organization_id=org_id, user_id=user_id) as db:
                user = await self._user(db, user_id)
                bundle = await self._bundle(db, user, org_id, agent_id, version_id)
                initial = await asyncio.to_thread(
                    lambda: RuntimeEngine(
                        version_id,
                        bundle.snapshot["config"],
                        lambda _: Prediction(version_id, None),
                    ).initial_state()
                )
                reserved = await reserve_initial_turn(
                    db,
                    user,
                    org_id,
                    agent_id,
                    version_id,
                    initial,
                    self.settings,
                    idempotency_key=idempotency_key,
                    http_mode=http_mode,
                )
            replayed = isinstance(reserved, ReplayedStart)
            if isinstance(reserved, ReplayedStart):
                saved = reserved.result
            else:
                live_http = (
                    LiveHttpTurn(org_id, user_id, reserved.credential, reserved.turn, deadline)
                    if http_mode == "live"
                    else None
                )
                result = await self._execute_in_slot("start", bundle, user_id, live_http=live_http)
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    saved = await complete_initial_turn(
                        db, await self._user(db, user_id), org_id, reserved, result, self.settings
                    )
            return ChatResponse(
                reserved.conversation_id,
                saved.revision,
                saved.status,
                saved.messages,
                saved.trace,
                saved.error_code,
                reserved.credential,
                saved.turn_id,
                replayed=replayed,
                replay_available=saved.replay_available,
            )

        try:
            async with asyncio.timeout(20):
                return await run_limited(self.factory, work)
        except TimeoutError:
            raise ExecutionError("runtime_request_timeout") from None

    async def _run_detached(self, operation: Coroutine[Any, Any, ChatResponse]) -> ChatResponse:
        if self._closing:
            operation.close()
            raise HTTPException(503, "Тестовый чат завершает работу")
        task = asyncio.create_task(operation)
        self._pending.add(task)

        def finished(completed: asyncio.Task[ChatResponse]) -> None:
            self._pending.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def send(
        self,
        user_id: UUID,
        org_id: UUID,
        conversation_id: UUID,
        revision: int,
        text: str,
        credential: str,
        message_id: UUID,
        idempotency_key: str,
    ) -> ChatResponse:
        return await self._run_detached(
            self._send(
                user_id,
                org_id,
                conversation_id,
                revision,
                text,
                credential,
                message_id,
                idempotency_key,
            )
        )

    async def _send(
        self,
        user_id: UUID,
        org_id: UUID,
        conversation_id: UUID,
        revision: int,
        text: str,
        credential: str,
        message_id: UUID,
        idempotency_key: str,
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
                    reserved = await reserve_message(
                        db,
                        user,
                        org_id,
                        conversation_id,
                        credential,
                        revision,
                        message_id,
                        idempotency_key,
                        text,
                        self.settings,
                    )
                if isinstance(reserved, Replay):
                    return ChatResponse(
                        conversation_id,
                        reserved.revision,
                        reserved.status,
                        reserved.messages,
                        reserved.trace,
                        reserved.error_code,
                        turn_id=reserved.turn_id,
                        replayed=True,
                        replay_available=reserved.replay_available,
                    )
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    await self._user(db, user_id)
                    row = await db.get(TestConversation, conversation_id)
                    if row is None or row.created_by != user_id:
                        raise HTTPException(404, "Диалог не найден")
                    agent_id, version_id, http_mode = row.agent_id, row.version_id, row.http_mode
                live_http = (
                    LiveHttpTurn(
                        org_id,
                        user_id,
                        credential,
                        reserved,
                        asyncio.get_running_loop().time() + 20,
                    )
                    if http_mode == "live"
                    else None
                )
                result = await self._execute(
                    "send",
                    user_id,
                    org_id,
                    agent_id,
                    version_id,
                    reserved.reservation.state,
                    text,
                    live_http,
                )
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=user_id
                ) as db:
                    saved = await complete_message(
                        db,
                        await self._user(db, user_id),
                        org_id,
                        conversation_id,
                        credential,
                        reserved,
                        result,
                        self.settings,
                    )
                return ChatResponse(
                    conversation_id,
                    saved.revision,
                    saved.status,
                    saved.messages,
                    saved.trace,
                    saved.error_code,
                    turn_id=saved.turn_id,
                    replay_available=saved.replay_available,
                )
        except TimeoutError:
            raise ExecutionError("runtime_request_timeout") from None
