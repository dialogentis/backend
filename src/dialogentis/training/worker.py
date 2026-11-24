"""Prepare and store an authenticated scenario version."""

from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import ArtifactQuotaExceeded, reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError
from dialogentis.jobs.models import Job
from dialogentis.nlu.model_store import ModelIdentity, pack_dialogue_manifest
from dialogentis.versions.models import AgentVersion
from dialogentis.versions.prepare import canonical_hash, nlu_hash

STAGES = {"validation", "packaging"}


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
        if canonical_hash(snapshot) != config_hash or nlu_hash(snapshot) != expected_nlu_hash:
            raise ValueError("Snapshot hash mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        raise PermanentJobError("training_snapshot_invalid") from exc
    await set_stage(factory, current, "validation")
    if not snapshot["llm_model"]:
        raise PermanentJobError("training_snapshot_invalid")
    await set_stage(factory, current, "packaging")
    model = pack_dialogue_manifest(identity, config_hash, settings)
    report = {"version_id": str(identity.version_id), "graph_validation": "passed"}

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
