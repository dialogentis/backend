import json
import os
import tempfile
from pathlib import Path
from uuid import UUID


def write_marker(root: Path, organization_id: UUID) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(
        {"version": 1, "organization_id": str(organization_id)}, separators=(",", ":")
    )
    descriptor, temp_name = tempfile.mkstemp(prefix=".deletion-", dir=root)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, root / f"{organization_id}.json")
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def read_markers(root: Path) -> list[UUID]:
    if not root.is_dir():
        raise ValueError("Deletion journal is unavailable")
    result = []
    for path in root.glob("*.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid deletion journal entry")
        value = json.loads(path.read_text())
        if set(value) != {"version", "organization_id"} or value["version"] != 1:
            raise ValueError("Invalid deletion journal entry")
        organization_id = UUID(value["organization_id"])
        if path.name != f"{organization_id}.json":
            raise ValueError("Invalid deletion journal entry")
        result.append(organization_id)
    return result
