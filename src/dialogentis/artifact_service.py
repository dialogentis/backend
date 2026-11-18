"""Tenant-scoped storage operations; callers must check the user's authorization first."""

import asyncio
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.artifact_records import Artifact
from dialogentis.artifacts import ArtifactInfo, ArtifactStorage
from dialogentis.config import Settings
from dialogentis.datasets.export_models import ExportTask
from dialogentis.datasets.import_models import ImportTask
from dialogentis.jobs.models import Job
from dialogentis.models import Organization
from dialogentis.runtime.history_export_models import HistoryExportTask


class ArtifactUnavailable(ValueError):
    pass


class ArtifactQuotaExceeded(ValueError):
    pass


async def _now(db: AsyncSession):
    now = await db.scalar(select(func.clock_timestamp()))
    assert now is not None
    return now


async def _active_organization(db: AsyncSession, organization_id: UUID) -> None:
    org = await db.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    if org is None or org.deletion_requested_at is not None:
        raise ArtifactUnavailable("Organization unavailable")


async def _blocking[T](operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError
    return result


async def reserve_artifact(
    db: AsyncSession,
    organization_id: UUID,
    *,
    kind: Literal["temporary", "model"],
    max_bytes: int,
    settings: Settings,
    job_id: UUID | None = None,
    ttl_seconds: int = 3600,
    artifact_id: UUID | None = None,
) -> Artifact:
    if (
        kind not in {"temporary", "model"}
        or type(max_bytes) is not int
        or max_bytes < 0
        or (artifact_id is not None and not isinstance(artifact_id, UUID))
    ):
        raise ValueError("Invalid artifact reservation")
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 3600:
        raise ValueError("Invalid artifact TTL")
    if job_id is not None:
        job = await db.scalar(
            select(Job)
            .where(Job.organization_id == organization_id, Job.id == job_id)
            .with_for_update()
        )
        if job is None:
            raise ArtifactUnavailable("Job unavailable")
    await _active_organization(db, organization_id)
    used = await db.scalar(
        select(func.coalesce(func.sum(Artifact.size_bytes), 0)).where(
            Artifact.organization_id == organization_id, Artifact.kind == kind
        )
    )
    quota = (
        settings.artifact_quota_bytes
        if kind == "model"
        else settings.temporary_artifact_quota_bytes
    )
    if int(used or 0) + max_bytes > quota:
        raise ArtifactQuotaExceeded("Artifact quota exceeded")
    now = await _now(db)
    artifact = Artifact(
        id=artifact_id or uuid4(),
        organization_id=organization_id,
        job_id=job_id,
        kind=kind,
        status="reserved",
        size_bytes=max_bytes,
        expires_at=now + timedelta(seconds=ttl_seconds),
    )
    db.add(artifact)
    await db.flush()
    return artifact


async def _locked_artifact(db: AsyncSession, organization_id: UUID, artifact_id: UUID) -> Artifact:
    await _active_organization(db, organization_id)
    artifact = await db.scalar(
        select(Artifact)
        .where(Artifact.organization_id == organization_id, Artifact.id == artifact_id)
        .with_for_update()
    )
    if artifact is None:
        raise ArtifactUnavailable("Artifact unavailable")
    return artifact


async def write_artifact(
    db: AsyncSession,
    storage: ArtifactStorage,
    organization_id: UUID,
    artifact_id: UUID,
    chunks: Iterable[bytes],
    *,
    expected_kind: Literal["temporary", "model"] | None = None,
) -> Artifact:
    artifact = await _locked_artifact(db, organization_id, artifact_id)
    now = await _now(db)
    if (
        (expected_kind is not None and artifact.kind != expected_kind)
        or artifact.status != "reserved"
        or artifact.expires_at is None
        or artifact.expires_at <= now
    ):
        raise ArtifactUnavailable("Artifact reservation unavailable")
    max_bytes = artifact.size_bytes

    def write() -> ArtifactInfo:
        storage.delete(organization_id, artifact_id)
        return storage.write(organization_id, artifact_id, chunks, max_bytes=max_bytes)

    info = await _blocking(write)
    artifact.status, artifact.size_bytes, artifact.sha256 = "ready", info.size, info.sha256
    if artifact.kind == "model":
        artifact.expires_at = None
    await db.flush()
    return artifact


async def read_artifact(
    db: AsyncSession,
    storage: ArtifactStorage,
    organization_id: UUID,
    artifact_id: UUID,
    *,
    max_bytes: int,
    expected_kind: Literal["temporary", "model"] | None = None,
) -> bytes:
    artifact = await _locked_artifact(db, organization_id, artifact_id)
    now = await _now(db)
    if (
        (expected_kind is not None and artifact.kind != expected_kind)
        or artifact.status != "ready"
        or (artifact.expires_at is not None and artifact.expires_at <= now)
    ):
        raise ArtifactUnavailable("Artifact unavailable")
    assert artifact.sha256 is not None
    info = ArtifactInfo(size=artifact.size_bytes, sha256=artifact.sha256)
    return await _blocking(
        lambda: storage.read(organization_id, artifact_id, info, max_bytes=max_bytes)
    )


async def cleanup_expired(db: AsyncSession, storage: ArtifactStorage, organization_id: UUID) -> int:
    await _active_organization(db, organization_id)
    now = await _now(db)
    expired = (
        await db.scalars(
            select(Artifact)
            .where(Artifact.organization_id == organization_id, Artifact.expires_at <= now)
            .order_by(Artifact.expires_at)
            .limit(100)
            .with_for_update()
        )
    ).all()
    for artifact in expired:
        for task in (ExportTask, HistoryExportTask):
            await db.execute(
                update(task)
                .where(task.organization_id == organization_id, task.artifact_id == artifact.id)
                .values(artifact_id=None)
            )
        await db.execute(
            update(ImportTask)
            .where(
                ImportTask.organization_id == organization_id,
                ImportTask.input_artifact_id == artifact.id,
            )
            .values(input_artifact_id=None)
        )
        await _blocking(
            lambda artifact_id=artifact.id: storage.delete(organization_id, artifact_id)
        )
        await db.delete(artifact)
    await db.flush()
    return len(expired)


async def cleanup_orphans(db: AsyncSession, storage: ArtifactStorage, organization_id: UUID) -> int:
    await _active_organization(db, organization_id)
    cutoff = await _now(db) - timedelta(hours=1)
    identifiers = set(
        (
            await db.scalars(select(Artifact.id).where(Artifact.organization_id == organization_id))
        ).all()
    )
    return await _blocking(
        lambda: storage.prune_orphans(organization_id, identifiers, older_than=cutoff)
    )
