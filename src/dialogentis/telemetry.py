import json
import logging
import re
from contextvars import ContextVar
from datetime import UTC, datetime
from time import monotonic
from uuid import UUID

request_id: ContextVar[UUID | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("dialogentis.events")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
CODE = re.compile(r"[a-z][a-z0-9_]{0,79}")


def log_event(
    service: str,
    event: str,
    *,
    started: float,
    turn_id: UUID | None = None,
    job_id: UUID | None = None,
    status: int | None = None,
    error_code: str | None = None,
    operation: str | None = None,
) -> None:
    level = (
        logging.WARNING if error_code or (status is not None and status >= 500) else logging.INFO
    )
    record: dict[str, str | int] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": logging.getLevelName(level).lower(),
        "service": service,
        "event": event,
        "duration_ms": max(0, round((monotonic() - started) * 1000)),
    }
    for key, value in (("request_id", request_id.get()), ("turn_id", turn_id), ("job_id", job_id)):
        if value is not None:
            record[key] = str(value)
    if status is not None:
        record["status"] = status
    if operation is not None:
        record["operation"] = operation if CODE.fullmatch(operation) else "unmatched"
    if error_code is not None:
        record["error_code"] = error_code if CODE.fullmatch(error_code) else "internal_error"
    logger.log(level, json.dumps(record, separators=(",", ":")))
