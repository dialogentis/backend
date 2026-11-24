import asyncio
from datetime import timedelta

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_service import ArtifactQuotaExceeded, reserve_artifact, write_artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import get_settings
from dialogentis.datasets.csv_export import export_csv
from dialogentis.datasets.export_models import ExportTask
from dialogentis.datasets.json_transfer import PortableConfig, export_json
from dialogentis.datasets.schema import Dataset
from dialogentis.db import unit_of_work
from dialogentis.jobs.lifecycle import Apply, Claim, PermanentJobError, RetryableJobError
from dialogentis.jobs.models import Job


def serialize_export(config: dict, format: str) -> bytes:
    try:
        if format == "json":
            portable = {
                key: value
                for key, value in config.items()
                if key not in {"nlu_settings", "quality_settings"}
            }
            return export_json(PortableConfig.model_validate(portable))
        if format == "csv":
            return export_csv(
                Dataset.model_validate(
                    {key: config.get(key) for key in ("intents", "entities", "examples")}
                )
            )
    except (ValidationError, ValueError, UnicodeError) as exc:
        raise PermanentJobError("invalid_export_configuration") from exc
    raise PermanentJobError("invalid_export_format")


async def prepare_export(factory: async_sessionmaker[AsyncSession], current: Claim) -> Apply:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        task = await db.get(ExportTask, current.job_id)
        now = await db.scalar(select(func.clock_timestamp()))
        if task is None or task.snapshot is None or now is None or task.snapshot_expires_at <= now:
            raise PermanentJobError("export_expired")
        snapshot, format = task.snapshot, task.format
    content = await asyncio.to_thread(serialize_export, snapshot, format)

    async def apply(db: AsyncSession, job: Job) -> None:
        task = await db.scalar(
            select(ExportTask).where(ExportTask.job_id == job.id).with_for_update()
        )
        now = await db.scalar(select(func.clock_timestamp()))
        if task is None or task.snapshot is None or now is None or task.snapshot_expires_at <= now:
            raise PermanentJobError("export_expired")
        settings = get_settings()
        try:
            artifact = await reserve_artifact(
                db,
                current.organization_id,
                kind="temporary",
                max_bytes=len(content),
                settings=settings,
                job_id=job.id,
            )
            await write_artifact(
                db,
                ArtifactStorage(settings.artifact_root),
                current.organization_id,
                artifact.id,
                [content],
            )
        except ArtifactQuotaExceeded as exc:
            raise PermanentJobError("artifact_quota_exceeded") from exc
        except OSError as exc:
            raise RetryableJobError("Artifact storage unavailable") from exc
        completed = await db.scalar(select(func.clock_timestamp()))
        assert completed is not None
        artifact.expires_at = completed + timedelta(hours=1)
        task.artifact_id = artifact.id
        task.download_until = completed + timedelta(minutes=15)
        task.snapshot = None

    return apply


async def scrub_exports(db: AsyncSession, organization_id) -> None:
    terminal = select(Job.id).where(
        Job.organization_id == organization_id, Job.status.in_(["failed", "cancelled", "succeeded"])
    )
    await db.execute(
        update(ExportTask)
        .where(
            ExportTask.organization_id == organization_id,
            ExportTask.snapshot.is_not(None),
            (ExportTask.snapshot_expires_at <= func.clock_timestamp())
            | ExportTask.job_id.in_(terminal),
        )
        .values(snapshot=None)
    )
