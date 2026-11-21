import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class HistoryMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: UUID
    job_id: UUID
    requested_at: AwareDatetime
    conversation_ids: list[UUID] = Field(min_length=1)

    @property
    def filename(self) -> str:
        return f"history-v1-{self.job_id}.json"


class HistoryPolicyMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: UUID
    revision: int = Field(ge=1)
    retention_days: int = Field(ge=1, le=30)
    store_message_text: bool
    retention_cutoff: AwareDatetime | None
    purge_text: bool

    @model_validator(mode="after")
    def require_cutoff_for_purge(self):
        if self.purge_text and self.retention_cutoff is None:
            raise ValueError("Missing retention cutoff for text purge")
        return self

    @property
    def filename(self) -> str:
        return f"history-policy-v1-{self.organization_id}-{self.revision}.json"


def _write_entry(root: Path, filename: str, payload: str) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temp_name = tempfile.mkstemp(prefix=".deletion-", dir=root)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, root / filename)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def write_marker(root: Path, organization_id: UUID) -> None:
    payload = json.dumps(
        {"version": 1, "organization_id": str(organization_id)}, separators=(",", ":")
    )
    _write_entry(root, f"{organization_id}.json", payload)


def write_history_marker(root: Path, marker: HistoryMarker) -> None:
    _write_entry(root, marker.filename, marker.model_dump_json())


def write_history_policy(root: Path, marker: HistoryPolicyMarker) -> None:
    _write_entry(root, marker.filename, marker.model_dump_json())


def _read_entry(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Invalid deletion journal entry")
    return path.read_text()


def read_markers(root: Path) -> list[UUID]:
    if not root.is_dir():
        raise ValueError("Deletion journal is unavailable")
    result = []
    for path in root.glob("*.json"):
        if path.name.startswith(("history-v1-", "history-policy-v1-")):
            continue
        value = json.loads(_read_entry(path))
        if set(value) != {"version", "organization_id"} or value["version"] != 1:
            raise ValueError("Invalid deletion journal entry")
        organization_id = UUID(value["organization_id"])
        if path.name != f"{organization_id}.json":
            raise ValueError("Invalid deletion journal entry")
        result.append(organization_id)
    return result


def read_history_markers(root: Path) -> list[HistoryMarker]:
    if not root.is_dir():
        raise ValueError("Deletion journal is unavailable")
    markers = []
    for path in root.glob("history-v1-*.json"):
        marker = HistoryMarker.model_validate_json(_read_entry(path))
        if path.name != marker.filename:
            raise ValueError("Invalid history deletion journal entry")
        markers.append(marker)
    return markers


def read_history_policies(root: Path) -> list[HistoryPolicyMarker]:
    if not root.is_dir():
        raise ValueError("Deletion journal is unavailable")
    markers = []
    for path in root.glob("history-policy-v1-*.json"):
        marker = HistoryPolicyMarker.model_validate_json(_read_entry(path))
        if path.name != marker.filename:
            raise ValueError("Invalid history policy journal entry")
        markers.append(marker)
    return markers
