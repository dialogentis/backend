"""Internal orchestration of idempotent test-chat turns."""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import read_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.auth.security import digest
from dialogentis.channels.api_access import ApiAccess
from dialogentis.channels.widget_access import WidgetAccess
from dialogentis.config import Settings
from dialogentis.datasets.validation import Issue
from dialogentis.db import unit_of_work
from dialogentis.flow_validation import validate_flow
from dialogentis.http_action import HttpActionConfig, MockIntegration, mock_http_action
from dialogentis.integrations.models import Integration
from dialogentis.jobs.models import Job
from dialogentis.models import Agent, Organization, User
from dialogentis.nlu.dialogue import DialogueDecision
from dialogentis.nlu.model_store import ModelIdentity
from dialogentis.runtime.conversations import Principal, PublicAccess, _access
from dialogentis.runtime.core import (
    RuntimeEngine,
    RuntimeResult,
    RuntimeState,
    TraceEvent,
)
from dialogentis.runtime.executor import (
    ExecutionError,
    HistoryLoader,
    HttpHandler,
    RuntimeExecutor,
    RuntimeTarget,
)
from dialogentis.runtime.history import load_dialogue_context
from dialogentis.runtime.history_context import HistoryContext
from dialogentis.runtime.idempotency import (
    Replay,
    ReplayedStart,
    complete_initial_turn,
    complete_message,
    reserve_initial_turn,
    reserve_message,
)
from dialogentis.runtime.limits import SLOT_COUNT, run_limited
from dialogentis.runtime.live_http import LiveHttpTurn
from dialogentis.runtime.models import TestConversation
from dialogentis.runtime.rate_limits import limit_api_turn, limit_widget_turn
from dialogentis.runtime.receipts import start_identity
from dialogentis.slots import validate_slots
from dialogentis.telemetry import log_event
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import version_status
from dialogentis.versions.release_check import ReleasePin
from dialogentis.versions.release_models import Release

type Actor = UUID | PublicAccess
type Manifest = dict[tuple[UUID, UUID], ReleasePin]


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
    release_id: UUID | None = None
    expires_at: datetime | None = None


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
        self._runtime_slots = asyncio.BoundedSemaphore(SLOT_COUNT)
        self._pending: set[asyncio.Task[ChatResponse]] = set()
        self._closing = False

    async def shutdown(self, grace_period: float = 30) -> None:
        self._closing = True
        try:
            if self._pending:
                _, pending = await asyncio.wait(self._pending, timeout=grace_period)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await self.executor.close()

    async def _user(self, db: AsyncSession, user_id: Actor) -> Principal:
        if isinstance(user_id, PublicAccess):
            return user_id
        user = await db.get(User, user_id)
        if user is None or not user.email_verified:
            raise HTTPException(403, "Подтвердите адрес почты")
        return user

    async def _bundle(
        self,
        db: AsyncSession,
        user: Principal,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID,
        manifest: Manifest | None = None,
    ) -> _Bundle:
        await _access(db, user, org_id)
        pin = None
        if manifest is not None:
            pin = manifest.get((agent_id, version_id))
            if pin is None:
                raise HTTPException(404, "Версия отсутствует в публикации")
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
        status = version_status(version, job)
        if status == "obsolete":
            raise HTTPException(409, "Эта версия устарела. Подготовьте новую версию агента.")
        if status != "ready" or version.model_artifact_id is None:
            raise HTTPException(409, "Версия агента ещё не готова")
        if pin and (
            pin.model_artifact_id != version.model_artifact_id
            or pin.config_hash != version.config_hash
        ):
            raise ExecutionError("runtime_invalid_request")
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
        user_id: Actor,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID,
        conversation_id: UUID,
        state: RuntimeState,
        text: str,
        live_http: LiveHttpTurn | None = None,
        manifest: Manifest | None = None,
    ) -> RuntimeResult:
        async def load_history() -> HistoryContext:
            async with unit_of_work(
                self.factory, organization_id=org_id, user_id=session_user(user_id)
            ) as db:
                user = await self._user(db, user_id)
                if isinstance(user, PublicAccess):
                    await user.authorize(db)
                else:
                    await _access(db, user, org_id)
                org = await db.get(Organization, org_id)
                assert org is not None
                return await load_dialogue_context(
                    db, org, conversation_id, state.turn + 1, self.settings
                )

        async def work() -> RuntimeResult:
            async with unit_of_work(
                self.factory, organization_id=org_id, user_id=session_user(user_id)
            ) as db:
                bundle = await self._bundle(
                    db, await self._user(db, user_id), org_id, agent_id, version_id, manifest
                )
            return await self._execute_in_slot(
                "send", bundle, user_id, state, text, live_http, manifest, load_history
            )

        try:
            async with asyncio.timeout(20), self._runtime_slots:
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
        user_id: Actor,
        state=None,
        text=None,
        live_http: LiveHttpTurn | None = None,
        manifest: Manifest | None = None,
        history_loader: HistoryLoader | None = None,
    ) -> RuntimeResult:
        async def load_target(agent_id: UUID, version_id: UUID) -> RuntimeTarget | None:
            org_id = bundle.identity.organization_id
            try:
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=session_user(user_id)
                ) as db:
                    target = await self._bundle(
                        db, await self._user(db, user_id), org_id, agent_id, version_id, manifest
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
            history_loader=history_loader,
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

    async def start_public(self, access: PublicAccess, idempotency_key: str) -> ChatResponse:
        if isinstance(access, WidgetAccess):
            idempotency_key = digest(f"{access.origin}:{idempotency_key}")
        return await self._run_detached(
            self._start(
                access, access.organization_id, access.agent_id, None, idempotency_key, "live"
            )
        )

    async def _start(
        self,
        user_id: Actor,
        org_id: UUID,
        agent_id: UUID,
        version_id: UUID | None,
        idempotency_key: str | None,
        http_mode: str,
    ) -> ChatResponse:
        started = monotonic()
        deadline = asyncio.get_running_loop().time() + 20

        async def work() -> ChatResponse:
            async with unit_of_work(
                self.factory, organization_id=org_id, user_id=session_user(user_id)
            ) as db:
                user = await self._user(db, user_id)
                selected = version_id
                release_id = None
                manifest = None
                existing = None
                if isinstance(user, PublicAccess):
                    await user.authorize(db)
                    assert idempotency_key is not None
                    conversation_id, _, _ = start_identity(
                        self.settings, org_id, user.id, agent_id, idempotency_key
                    )
                    existing = await db.get(TestConversation, conversation_id)
                else:
                    await _access(db, user, org_id)
                agent = await db.get(Agent, agent_id)
                if agent is None or (agent.status != "active" and existing is None):
                    raise HTTPException(409, "Архивный агент недоступен для нового диалога")
                if isinstance(user, PublicAccess):
                    release_id = existing.release_id if existing else agent.current_release_id
                    release, manifest = await load_release(db, org_id, agent_id, release_id)
                    selected = release.root_version_id
                if isinstance(user, WidgetAccess) and existing is None:
                    await limit_widget_turn(db, user, start=True)
                assert selected is not None
                bundle = await self._bundle(db, user, org_id, agent_id, selected, manifest)
                initial = await asyncio.to_thread(
                    lambda: RuntimeEngine(
                        selected,
                        bundle.snapshot["config"],
                        lambda _: DialogueDecision(action="clarify", intent=None, values=()),
                    ).initial_state()
                )
                reserved = await reserve_initial_turn(
                    db,
                    user,
                    org_id,
                    agent_id,
                    selected,
                    initial,
                    self.settings,
                    idempotency_key=idempotency_key,
                    http_mode=http_mode,
                    release_id=release_id,
                )
                if isinstance(user, ApiAccess) and not isinstance(reserved, ReplayedStart):
                    await limit_api_turn(db, org_id, user.id, start=True)
            replayed = isinstance(reserved, ReplayedStart)
            if isinstance(reserved, ReplayedStart):
                saved = reserved.result
            else:
                log_event(
                    "runtime", "turn_accepted", started=started, turn_id=reserved.turn.turn_id
                )
                live_http = (
                    LiveHttpTurn(org_id, user_id, reserved.credential, reserved.turn, deadline)
                    if http_mode == "live"
                    else None
                )
                result = await self._execute_in_slot(
                    "start", bundle, user_id, live_http=live_http, manifest=manifest
                )
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=session_user(user_id)
                ) as db:
                    saved = await complete_initial_turn(
                        db, await self._user(db, user_id), org_id, reserved, result, self.settings
                    )
            log_event(
                "runtime",
                "turn_replayed" if replayed else "turn_finished",
                started=started,
                turn_id=saved.turn_id,
                error_code=saved.error_code,
            )
            expires_at = None
            if release_id is not None:
                async with unit_of_work(self.factory, organization_id=org_id) as db:
                    row = await db.get(TestConversation, reserved.conversation_id)
                    if row is None:
                        raise HTTPException(404, "Диалог не найден")
                    expires_at = row.idle_expires_at
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
                release_id=release_id,
                expires_at=expires_at,
            )

        try:
            async with asyncio.timeout(20), self._runtime_slots:
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
        user_id: Actor,
        org_id: UUID,
        conversation_id: UUID,
        revision: int | None,
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
        user_id: Actor,
        org_id: UUID,
        conversation_id: UUID,
        revision: int | None,
        text: str,
        credential: str,
        message_id: UUID,
        idempotency_key: str,
    ) -> ChatResponse:
        started = monotonic()
        accepted_id = None
        if not isinstance(text, str) or not 1 <= len(text) <= 2000:
            raise ExecutionError("runtime_invalid_request")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise ExecutionError("runtime_invalid_request") from None
        try:
            async with asyncio.timeout(25):
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=session_user(user_id)
                ) as db:
                    user = await self._user(db, user_id)
                    if isinstance(user, PublicAccess):
                        await user.authorize(db)
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
                    if isinstance(user, ApiAccess) and not isinstance(reserved, Replay):
                        await limit_api_turn(db, org_id, user.id, start=False)
                    if isinstance(user, WidgetAccess) and not isinstance(reserved, Replay):
                        await limit_widget_turn(db, user, start=False)
                if isinstance(reserved, Replay):
                    log_event("runtime", "turn_replayed", started=started, turn_id=reserved.turn_id)
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
                accepted_id = reserved.turn_id
                log_event("runtime", "turn_accepted", started=started, turn_id=accepted_id)
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=session_user(user_id)
                ) as db:
                    await self._user(db, user_id)
                    row = await db.get(TestConversation, conversation_id)
                    if row is None:
                        raise HTTPException(404, "Диалог не найден")
                    agent_id, version_id, http_mode = row.agent_id, row.version_id, row.http_mode
                    manifest = None
                    if row.release_id is not None:
                        _, manifest = await load_release(db, org_id, agent_id, row.release_id)
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
                    user_id,
                    org_id,
                    agent_id,
                    version_id,
                    conversation_id,
                    reserved.reservation.state,
                    text,
                    live_http,
                    manifest,
                )
                async with unit_of_work(
                    self.factory, organization_id=org_id, user_id=session_user(user_id)
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
                        external_http_ms=round(live_http.external_http_seconds * 1000)
                        if live_http is not None
                        else 0,
                    )
                log_event(
                    "runtime",
                    "turn_finished",
                    started=started,
                    turn_id=saved.turn_id,
                    error_code=saved.error_code,
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
            log_event(
                "runtime",
                "turn_interrupted",
                started=started,
                turn_id=accepted_id,
                error_code="runtime_request_timeout",
            )
            raise ExecutionError("runtime_request_timeout") from None
        except ExecutionError as exc:
            log_event(
                "runtime",
                "turn_interrupted",
                started=started,
                turn_id=accepted_id,
                error_code=exc.code,
            )
            raise


def session_user(actor: Actor) -> UUID | None:
    return actor if isinstance(actor, UUID) else None


async def load_release(
    db: AsyncSession, org_id: UUID, agent_id: UUID, release_id: UUID | None
) -> tuple[Release, Manifest]:
    release = await db.scalar(
        select(Release).where(
            Release.id == release_id,
            Release.organization_id == org_id,
            Release.agent_id == agent_id,
        )
    )
    if release is None:
        raise HTTPException(409, "У агента нет доступной публикации")
    pins = [ReleasePin.model_validate(item) for item in release.dependency_manifest]
    return release, {(pin.agent_id, pin.version_id): pin for pin in pins}
