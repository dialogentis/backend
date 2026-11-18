import asyncio
import json
from datetime import timedelta
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.agents import agent_and_draft
from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import ArtifactUnavailable, read_artifact
from dialogentis.artifacts import ArtifactIntegrityError, ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.datasets.csv_import import MAX_FILE_BYTES, parse_csv, preview_csv
from dialogentis.datasets.import_models import ImportTask
from dialogentis.datasets.json_transfer import PortableConfig, parse_json, preview_json
from dialogentis.datasets.routes import validate_configuration
from dialogentis.datasets.schema import Dataset
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError, RetryableJobError
from dialogentis.jobs.models import Job
from dialogentis.slots import removed_slot_references


def analyze_import(
    content: bytes, format: Literal["csv", "json"], mode: Literal["merge", "replace"], config: dict
) -> tuple[dict | None, dict]:
    report: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "counts": {"intents": 0, "entities": 0, "examples": 0},
        "new_codes": {},
        "existing_codes": {},
        "codes_truncated": False,
    }
    try:
        if format == "csv":
            parsed = parse_csv(content)
            report["counts"]["examples"] = len(parsed.rows)
            current = Dataset.model_validate(
                {key: config.get(key) for key in ("intents", "entities", "examples")}
            )
            preview = preview_csv(parsed, current, mode)
            report["errors"] = [
                {"row": issue.row, "field": issue.field, "path": [], "code": issue.code}
                for issue in preview.errors
            ]
            candidate = (
                {**config, **preview.dataset.model_dump(mode="json")}
                if preview.dataset is not None
                else None
            )
        else:
            imported, errors = parse_json(content)
            if errors:
                report["errors"] = [
                    {"row": None, "field": "file", "path": list(issue.path), "code": issue.code}
                    for issue in errors
                ]
                return None, report
            assert imported is not None
            current = PortableConfig.model_validate(config)
            for key in ("intents", "entities", "examples"):
                report["counts"][key] = len(getattr(imported, key))
            for key in ("intents", "entities"):
                old = {item.code for item in getattr(current, key)}
                incoming = {item.code for item in getattr(imported, key)}
                if len(incoming - old) > 1000 or len(incoming & old) > 1000:
                    report["codes_truncated"] = True
                report["new_codes"][key] = sorted(incoming - old)[:1000]
                report["existing_codes"][key] = sorted(incoming & old)[:1000]
            preview = preview_json(imported, current, mode)
            report["errors"] = [
                {"row": None, "field": "file", "path": list(issue.path), "code": issue.code}
                for issue in preview.errors
            ]
            report["warnings"] = [
                {"row": None, "field": "file", "path": list(issue.path), "code": issue.code}
                for issue in preview.warnings
            ]
            candidate = (
                preview.config.model_dump(mode="json") if preview.config is not None else None
            )
        if candidate is not None:
            config_errors, _ = validate_configuration(candidate)
            invalid_references = [
                item
                for item in config_errors
                if item.code
                in {
                    "unknown_intent_output",
                    "unknown_condition_slot",
                    "unknown_set_slot",
                    "unknown_ask_slot",
                    "unknown_ask_entity",
                    "invalid_http_action",
                    "invalid_handoff_config",
                    "invalid_slots",
                    "invalid_slot",
                    "duplicate_slot_id",
                    "duplicate_slot_code",
                }
            ]
            invalid_references.extend(removed_slot_references(config, candidate))
            if invalid_references:
                report["errors"] = [
                    *report["errors"],
                    *(
                        {"row": None, "field": "file", "path": list(item.path), "code": item.code}
                        for item in invalid_references[:1000]
                    ),
                ]
                candidate = None
        if (
            candidate is not None
            and len(json.dumps(candidate, ensure_ascii=False, allow_nan=False).encode())
            > MAX_FILE_BYTES
        ):
            report["errors"] = [
                *report["errors"],
                {"row": None, "field": "file", "path": [], "code": "configuration_too_large"},
            ]
            candidate = None
        return candidate, report
    except ValidationError, ValueError, UnicodeError:
        report["errors"] = [
            *report["errors"],
            {"row": None, "field": "file", "path": [], "code": "invalid_current_configuration"},
        ]
        return None, report


async def prepare_import(factory: async_sessionmaker[AsyncSession], current: Claim) -> Apply:
    settings = get_settings()
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        task = await db.get(ImportTask, current.job_id)
        now = await db.scalar(select(func.clock_timestamp()))
        if task is None or task.input_artifact_id is None or now is None or task.expires_at <= now:
            raise PermanentJobError("import_expired")
        _, draft = await agent_and_draft(db, current.organization_id, task.agent_id)
        if draft.revision != task.base_revision:
            raise PermanentJobError("import_stale")
        try:
            content = await read_artifact(
                db,
                ArtifactStorage(settings.artifact_root),
                current.organization_id,
                task.input_artifact_id,
                max_bytes=MAX_FILE_BYTES,
            )
        except ArtifactUnavailable as exc:
            raise PermanentJobError("import_expired") from exc
        except ArtifactIntegrityError as exc:
            raise PermanentJobError("import_file_invalid") from exc
        except OSError as exc:
            raise RetryableJobError("Artifact storage unavailable") from exc
        config, format, mode = draft.configuration, task.format, task.mode
    candidate, report = await asyncio.to_thread(analyze_import, content, format, mode, config)

    async def apply(db: AsyncSession, job: Job) -> None:
        task = await db.scalar(
            select(ImportTask).where(ImportTask.job_id == job.id).with_for_update()
        )
        now = await db.scalar(select(func.clock_timestamp()))
        if task is None or task.input_artifact_id is None or now is None or task.expires_at <= now:
            raise PermanentJobError("import_expired")
        _, draft = await agent_and_draft(db, current.organization_id, task.agent_id)
        if draft.revision != task.base_revision:
            raise PermanentJobError("import_stale")
        artifact = await db.get(Artifact, task.input_artifact_id)
        if artifact is None or artifact.expires_at is None or artifact.expires_at <= now:
            raise PermanentJobError("import_expired")
        task.candidate, task.report = candidate, report
        task.expires_at = artifact.expires_at = now + timedelta(hours=1)

    return apply


async def scrub_imports(db: AsyncSession, organization_id) -> None:
    terminal = select(Job.id).where(
        Job.organization_id == organization_id, Job.status.in_(["failed", "cancelled"])
    )
    await db.execute(
        update(ImportTask)
        .where(
            ImportTask.organization_id == organization_id,
            (ImportTask.expires_at <= func.clock_timestamp()) | ImportTask.job_id.in_(terminal),
        )
        .values(candidate=None, report=None)
    )
