"""Hard-bounded child-process execution for one trusted model and runtime turn."""

import asyncio
import json
import struct
import sys
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dialogentis.config import Settings
from dialogentis.nlu.model_store import ModelIdentity
from dialogentis.runtime.child import MAX_REQUEST, MAX_RESPONSE
from dialogentis.runtime.core import RuntimeResult, RuntimeState, TraceEvent


class ExecutionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class EventView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: UUID
    node_type: str = Field(max_length=32)
    event: str = Field(max_length=64)
    output: str | None = Field(default=None, max_length=128)
    fallback_reasons: tuple[str, ...] = Field(default=(), max_length=10)


class ResultView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: RuntimeState
    messages: tuple[str, ...] = Field(max_length=52)
    trace: tuple[EventView, ...] = Field(max_length=120)
    error_code: str | None = Field(default=None, max_length=64)


class RuntimeExecutor:
    def __init__(self, concurrency: int = 4, timeout: float = 20):
        if type(concurrency) is not int or concurrency < 1 or not 0 < timeout <= 20:
            raise ValueError("Invalid runtime execution limits")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = timeout

    async def execute(
        self,
        *,
        action: Literal["start", "send"],
        snapshot: dict,
        config_hash: str,
        identity: ModelIdentity,
        model: bytes,
        settings: Settings,
        state: RuntimeState | None = None,
        text: str | None = None,
    ) -> RuntimeResult:
        if settings.encryption_key is None:
            raise ExecutionError("runtime_key_unavailable")
        if (
            action not in ("start", "send")
            or (action == "start" and (state is not None or text is not None))
            or (
                action == "send"
                and (
                    not isinstance(state, RuntimeState)
                    or state.version_id != identity.version_id
                    or state.status != "waiting_input"
                    or not isinstance(text, str)
                    or not 1 <= len(text) <= 2000
                )
            )
        ):
            raise ExecutionError("runtime_invalid_request")
        if not isinstance(model, bytes) or not 1 <= len(model) <= min(
            settings.max_model_artifact_bytes, 2**32 - 1
        ):
            raise ExecutionError("runtime_invalid_model")
        try:
            request = json.dumps(
                {
                    "action": action,
                    "snapshot": snapshot,
                    "config_hash": config_hash,
                    "organization_id": str(identity.organization_id),
                    "agent_id": str(identity.agent_id),
                    "version_id": str(identity.version_id),
                    "artifact_id": str(identity.artifact_id),
                    "state": state.model_dump(mode="json") if state is not None else None,
                    "text": text,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except TypeError, ValueError, UnicodeError:
            raise ExecutionError("runtime_invalid_request") from None
        if not 1 <= len(request) <= MAX_REQUEST:
            raise ExecutionError("runtime_invalid_request")
        try:
            async with asyncio.timeout(self._timeout):
                async with self._semaphore:
                    return await self._run(request, model, identity, settings)
        except TimeoutError:
            raise ExecutionError("runtime_timeout") from None

    async def _run(
        self, request: bytes, model: bytes, identity: ModelIdentity, settings: Settings
    ) -> RuntimeResult:
        assert settings.encryption_key is not None
        env = {
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "DIALOGENTIS_RUNTIME_KEY": settings.encryption_key.get_secret_value(),
            "DIALOGENTIS_RUNTIME_KEY_VERSION": str(settings.encryption_key_version),
            "DIALOGENTIS_RUNTIME_MODEL_LIMIT": str(settings.max_model_artifact_bytes),
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dialogentis.runtime.child",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(struct.pack(">II", len(request), len(model)) + request)
            await process.stdin.drain()
            for offset in range(0, len(model), 65536):
                process.stdin.write(model[offset : offset + 65536])
                await process.stdin.drain()
            size = struct.unpack(">I", await process.stdout.readexactly(4))[0]
            if not 1 <= size <= MAX_RESPONSE:
                raise ExecutionError("runtime_invalid_response")
            output = await process.stdout.readexactly(size)
            if await process.wait() != 0:
                raise ExecutionError("runtime_failed")
            view = ResultView.model_validate_json(output)
            if view.state.version_id != identity.version_id:
                raise ExecutionError("runtime_invalid_response")
            return RuntimeResult(
                view.state,
                view.messages,
                tuple(TraceEvent(**event.model_dump()) for event in view.trace),
                view.error_code,
            )
        except ExecutionError:
            raise
        except (
            asyncio.IncompleteReadError,
            BrokenPipeError,
            ConnectionResetError,
            ValidationError,
            ValueError,
        ):
            raise ExecutionError("runtime_failed") from None
        finally:

            async def stop() -> None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()

            cleanup = asyncio.create_task(stop())
            interrupted = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    if cleanup.cancelled():
                        raise
                    interrupted = True
            if interrupted:
                raise asyncio.CancelledError
