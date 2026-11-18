"""Execute one runtime turn in an isolated process; stdin/stdout are private pipes."""

import json
import os
import queue
import struct
import sys
import threading
from dataclasses import asdict
from functools import lru_cache
from uuid import UUID

from pydantic import SecretStr

from dialogentis.config import Settings
from dialogentis.handoff import MAX_HANDOFF_DEPTH
from dialogentis.http_action import MockIntegration
from dialogentis.nlu.model_store import ModelIdentity, unpack_model
from dialogentis.runtime.core import RuntimeEngine, RuntimeState
from dialogentis.runtime.history_context import HistoryContext
from dialogentis.runtime.nlu import nlu_predictor
from dialogentis.versions.prepare import canonical_hash

MAX_REQUEST = 12 * 1024 * 1024
MAX_RESPONSE = 4 * 1024 * 1024


def _read_exact(size: int) -> bytes:
    data = sys.stdin.buffer.read(size)
    if len(data) != size:
        raise ValueError("Incomplete request")
    return data


def _watch_parent() -> None:
    if not os.read(sys.stdin.fileno(), 1):
        os._exit(73)


def _read_replies(replies: queue.Queue, model_limit: int) -> None:
    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            part = os.read(sys.stdin.fileno(), size - len(data))
            if not part:
                raise EOFError()
            data.extend(part)
        return bytes(data)

    try:
        while True:
            size, model_size = struct.unpack(">II", read_exact(8))
            if not 1 <= size <= MAX_REQUEST or not 0 <= model_size <= model_limit:
                raise ValueError("Invalid response size")
            parsed = json.loads(read_exact(size))
            replies.put((parsed, read_exact(model_size)))
    except Exception:
        os._exit(73)


def _write_frame(value: dict) -> None:
    output = json.dumps(
        value, ensure_ascii=False, allow_nan=False, default=str, separators=(",", ":")
    ).encode("utf-8")
    if not 1 <= len(output) <= MAX_RESPONSE:
        raise ValueError("Runtime output too large")
    sys.stdout.buffer.write(struct.pack(">I", len(output)) + output)
    sys.stdout.buffer.flush()


def main() -> None:
    request_size, model_size = struct.unpack(">II", _read_exact(8))
    model_limit = int(os.environ["DIALOGENTIS_RUNTIME_MODEL_LIMIT"])
    if not 1 <= request_size <= MAX_REQUEST or not 1 <= model_size <= model_limit:
        raise ValueError("Invalid request size")
    request = json.loads(_read_exact(request_size))
    model = _read_exact(model_size)
    replies: queue.Queue = queue.Queue(maxsize=1)
    handoff = request.get("handoff_bridge") is True
    historical = handoff and request.get("history_bridge") is True
    bridged = handoff or request.get("http_bridge") is True
    threading.Thread(
        target=_read_replies if bridged else _watch_parent,
        args=(replies, model_limit) if bridged else (),
        daemon=True,
    ).start()
    identity = ModelIdentity(
        *(
            UUID(request[key])
            for key in ("organization_id", "agent_id", "version_id", "artifact_id")
        )
    )
    settings = Settings(
        _env_file=None,
        encryption_key=SecretStr(os.environ["DIALOGENTIS_RUNTIME_KEY"]),
        encryption_key_version=int(os.environ["DIALOGENTIS_RUNTIME_KEY_VERSION"]),
        max_model_artifact_bytes=model_limit,
    )

    @lru_cache(maxsize=MAX_HANDOFF_DEPTH)
    def load_target(version_id: UUID) -> RuntimeEngine | None:
        _write_frame({"kind": "handoff_target", "version_id": str(version_id)})
        reply, target_model = replies.get()
        if reply == {"kind": "unavailable"} and not target_model:
            return None
        if reply["kind"] != "target" or UUID(reply["version_id"]) != version_id:
            raise ValueError("Invalid handoff target")
        target_identity = ModelIdentity(
            identity.organization_id,
            UUID(reply["agent_id"]),
            version_id,
            UUID(reply["artifact_id"]),
        )
        return make_engine(target_identity, reply, target_model)

    @lru_cache(maxsize=MAX_HANDOFF_DEPTH)
    def load_history(version_id: UUID, node_id: UUID) -> HistoryContext:
        _write_frame(
            {"kind": "handoff_history", "version_id": str(version_id), "node_id": str(node_id)}
        )
        reply, extra = replies.get()
        if extra:
            raise ValueError("Invalid history response")
        return HistoryContext.model_validate(reply)

    def make_engine(model_identity: ModelIdentity, payload: dict, artifact: bytes) -> RuntimeEngine:
        snapshot = payload["snapshot"]
        if snapshot["schema_version"] != 1 or canonical_hash(snapshot) != payload["config_hash"]:
            raise ValueError("Invalid snapshot")
        adapter = unpack_model(artifact, model_identity, settings)

        def http_handler(node_id: UUID, values: dict) -> tuple[str, dict]:
            _write_frame(
                {
                    "kind": "http_action",
                    "node_id": str(node_id),
                    "values": values,
                    "version_id": str(model_identity.version_id),
                }
            )
            reply, extra = replies.get()
            if extra or not isinstance(reply, dict) or set(reply) != {"outcome", "updates"}:
                raise ValueError("Invalid action response")
            return reply["outcome"], reply["updates"]

        mock_integrations = payload.get("mock_integrations")
        return RuntimeEngine(
            model_identity.version_id,
            snapshot["config"],
            nlu_predictor(adapter),
            mock_integrations=(
                [MockIntegration.model_validate(item) for item in mock_integrations]
                if mock_integrations is not None
                else None
            ),
            http_handler=http_handler if payload.get("http_bridge") is True else None,
            resolve_version=load_target if handoff else None,
            load_history=load_history if historical else None,
        )

    engine = make_engine(identity, request, model)
    if request["action"] == "start":
        result = engine.start()
    elif request["action"] == "send":
        result = engine.send(RuntimeState.model_validate(request["state"]), request["text"])
    else:
        raise ValueError("Unknown action")
    output = {
        "state": result.state.model_dump(mode="json"),
        "messages": result.messages,
        "history_input": result.history_input,
        "history_messages": result.history_messages,
        "trace": [asdict(event) for event in result.trace],
        "error_code": result.error_code,
    }
    _write_frame({"kind": "result", "result": output} if bridged else output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os._exit(1)
