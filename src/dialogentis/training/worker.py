"""Fenced training with a cancellable, isolated child process."""

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import ArtifactQuotaExceeded, reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings, get_settings
from dialogentis.datasets.schema import Dataset
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError
from dialogentis.jobs.models import Job
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.intent import training_data_hash as intent_training_hash
from dialogentis.nlu.learned_entities import training_data_hash as entity_training_hash
from dialogentis.nlu.model_store import ModelIdentity, unpack_model
from dialogentis.training.child import MAX_REPORT_BYTES, MAX_REQUEST_BYTES
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import canonical_hash, nlu_hash

STAGES = {"validation", "intent_training", "entity_training", "evaluation", "packaging"}


async def set_stage(factory: async_sessionmaker[AsyncSession], current: Claim, stage: str) -> None:
    if stage not in STAGES:
        raise PermanentJobError("training_invalid_stage")
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        result = await db.execute(
            update(Job)
            .where(
                Job.id == current.job_id,
                Job.organization_id == current.organization_id,
                Job.status == "running",
                Job.execution_token == current.token,
            )
            .values(stage=stage)
            .returning(Job.cancel_requested)
        )
        cancelled = result.scalar_one_or_none()
        if cancelled is None:
            raise PermanentJobError("training_fenced")
        if cancelled:
            raise PermanentJobError("training_cancelled")


async def run_child(
    request: dict, settings: Settings, progress: Callable[[str], Awaitable[None]]
) -> tuple[bytes, dict]:
    if settings.encryption_key is None:
        raise PermanentJobError("training_key_unavailable")
    encoded = (
        json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    if len(encoded) > MAX_REQUEST_BYTES:
        raise PermanentJobError("training_snapshot_too_large")
    with tempfile.TemporaryFile() as model_file, tempfile.TemporaryFile() as report_file:
        environment = {
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "DIALOGENTIS_TRAINING_KEY": settings.encryption_key.get_secret_value(),
            "DIALOGENTIS_TRAINING_KEY_VERSION": str(settings.encryption_key_version),
            "DIALOGENTIS_TRAINING_MODEL_LIMIT": str(settings.max_model_artifact_bytes),
            "DIALOGENTIS_TRAINING_MODEL_FD": str(model_file.fileno()),
            "DIALOGENTIS_TRAINING_REPORT_FD": str(report_file.fileno()),
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "dialogentis.training.child",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            pass_fds=(model_file.fileno(), report_file.fileno()),
        )

        async def stop() -> None:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    async with asyncio.timeout(3):
                        await process.wait()
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            if process.stdin is not None:
                process.stdin.close()

        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(encoded)
            await process.stdin.drain()
            async for line in process.stdout:
                try:
                    name = line.decode("ascii").strip()
                except UnicodeError as exc:
                    raise PermanentJobError("training_invalid_stage") from exc
                await progress(name)
            if await process.wait() != 0:
                raise PermanentJobError("training_failed")
            model_file.seek(0, os.SEEK_END)
            report_file.seek(0, os.SEEK_END)
            if (
                not 0 < model_file.tell() <= settings.max_model_artifact_bytes
                or not 0 < report_file.tell() <= MAX_REPORT_BYTES
            ):
                raise PermanentJobError("training_output_invalid")
            model_file.seek(0)
            report_file.seek(0)
            model = model_file.read()
            try:
                report = json.loads(report_file.read())
            except (ValueError, UnicodeError) as exc:
                raise PermanentJobError("training_output_invalid") from exc
            if not isinstance(report, dict):
                raise PermanentJobError("training_output_invalid")
            return model, report
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise PermanentJobError("training_failed") from exc
        finally:
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


def verify_output(
    model: bytes, report: dict, identity: ModelIdentity, snapshot: dict, settings: Settings
) -> None:
    adapter = unpack_model(model, identity, settings)
    config = snapshot["config"]
    dataset = Dataset.model_validate(
        {key: config[key] for key in ("intents", "entities", "examples")}
    )
    learned = any(item.method == "learned" for item in dataset.entities)
    if (
        adapter.intent_model.manifest.get("training_data_sha256") != intent_training_hash(dataset)
        or adapter.entities != dataset.entities
        or adapter.thresholds != Thresholds(**snapshot["nlu_settings"])
        or (adapter.entity_model is not None) != learned
        or (
            adapter.entity_model is not None
            and adapter.entity_model.manifest.get("training_data_sha256")
            != entity_training_hash(dataset)
        )
        or report.get("version_id") != str(identity.version_id)
        or report.get("graph_validation") != "not_checked"
        or not isinstance(report.get("intents"), dict)
        or not isinstance(report.get("entities"), dict)
    ):
        raise ValueError("Training output does not match snapshot")


async def prepare_training(factory: async_sessionmaker[AsyncSession], current: Claim) -> Apply:
    settings = get_settings()
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        version = await db.scalar(
            select(AgentVersion).where(
                AgentVersion.organization_id == current.organization_id,
                AgentVersion.job_id == current.job_id,
            )
        )
        if version is None or version.model_artifact_id is not None:
            raise PermanentJobError("training_version_missing")
        used = await db.scalar(
            select(func.coalesce(func.sum(Artifact.size_bytes), 0)).where(
                Artifact.organization_id == current.organization_id, Artifact.kind == "model"
            )
        )
        if int(used or 0) >= settings.artifact_quota_bytes:
            raise PermanentJobError("artifact_quota_exceeded")
        snapshot, config_hash, expected_nlu_hash = (
            version.snapshot,
            version.config_hash,
            version.nlu_hash,
        )
        identity = ModelIdentity(current.organization_id, version.agent_id, version.id, uuid4())
    try:
        if (
            canonical_hash(snapshot) != config_hash
            or nlu_hash(snapshot["config"], snapshot["nlu_settings"]) != expected_nlu_hash
        ):
            raise ValueError("Snapshot hash mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        raise PermanentJobError("training_snapshot_invalid") from exc
    await set_stage(factory, current, "validation")
    request = {
        "organization_id": str(identity.organization_id),
        "agent_id": str(identity.agent_id),
        "version_id": str(identity.version_id),
        "artifact_id": str(identity.artifact_id),
        "snapshot": snapshot,
    }
    model, report = await run_child(
        request, settings, lambda name: set_stage(factory, current, name)
    )
    try:
        await asyncio.to_thread(verify_output, model, report, identity, snapshot, settings)
    except Exception as exc:
        raise PermanentJobError("training_output_invalid") from exc

    async def apply(db: AsyncSession, job: Job) -> None:
        version = await db.scalar(
            select(AgentVersion)
            .where(
                AgentVersion.id == identity.version_id,
                AgentVersion.organization_id == current.organization_id,
                AgentVersion.job_id == job.id,
            )
            .with_for_update()
        )
        if version is None or version.model_artifact_id is not None:
            raise PermanentJobError("training_version_missing")
        try:
            artifact = await reserve_artifact(
                db,
                current.organization_id,
                kind="model",
                max_bytes=len(model),
                settings=settings,
                job_id=job.id,
                artifact_id=identity.artifact_id,
            )
            await write_artifact(
                db,
                ArtifactStorage(settings.artifact_root),
                current.organization_id,
                artifact.id,
                [model],
                expected_kind="model",
            )
        except ArtifactQuotaExceeded as exc:
            raise PermanentJobError("artifact_quota_exceeded") from exc
        except OSError as exc:
            raise PermanentJobError("artifact_storage_unavailable") from exc
        version.model_artifact_id = artifact.id
        version.validation_report = report

    return apply
