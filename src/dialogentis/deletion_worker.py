import asyncio
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.artifact_records import Artifact
from dialogentis.artifacts import ArtifactStorage
from dialogentis.auth.models import EmailOutbox
from dialogentis.config import get_settings
from dialogentis.datasets.export_models import ExportTask
from dialogentis.db import unit_of_work
from dialogentis.deletion_journal import write_marker
from dialogentis.jobs.lifecycle import Apply, Claim, RetryableJobError
from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.models import Agent, AgentDraft, Membership, Organization
from dialogentis.organizations import AuditEvent, Invitation, OrganizationDeletion


async def purge_tenant(
    db: AsyncSession, organization_id, *, expected_job_id=None, restoring=False
) -> None:
    org = await db.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    record = await db.scalar(
        select(OrganizationDeletion).where(OrganizationDeletion.organization_id == organization_id)
    )
    if expected_job_id is not None:
        if (
            org is None
            or org.deletion_requested_at is None
            or record is None
            or record.job_id != expected_job_id
        ):
            raise ValueError("Invalid deletion state")
    try:
        await asyncio.to_thread(
            ArtifactStorage(get_settings().artifact_root).purge_organization, organization_id
        )
    except OSError as exc:
        raise RetryableJobError("Artifact storage is unavailable") from exc
    keep_job_id = record.job_id if record is not None else None
    await db.execute(delete(ExportTask).where(ExportTask.organization_id == organization_id))
    await db.execute(delete(Artifact).where(Artifact.organization_id == organization_id))
    outbox = delete(OutboxEvent).where(OutboxEvent.organization_id == organization_id)
    jobs = delete(Job).where(Job.organization_id == organization_id)
    if keep_job_id is not None:
        outbox = outbox.where(OutboxEvent.job_id != keep_job_id)
        jobs = jobs.where(Job.id != keep_job_id)
    await db.execute(outbox)
    await db.execute(jobs)
    await db.execute(delete(EmailOutbox).where(EmailOutbox.organization_id == organization_id))
    for model in (AgentDraft, Agent, Invitation, AuditEvent, Membership):
        await db.execute(delete(model).where(model.organization_id == organization_id))
    if record is not None:
        if record.completed_at is None:
            record.completed_at = datetime.now(UTC)
        if restoring:
            job = await db.get(Job, record.job_id)
            if job is None:
                raise ValueError("Deletion job is missing")
            job.status = "succeeded"
            job.finished_at = record.completed_at
            job.execution_token = None
            job.next_retry_at = None
            job.error_code = None
    if org is not None:
        await db.delete(org)


async def prepare_deletion(factory: async_sessionmaker[AsyncSession], current: Claim) -> Apply:
    async with unit_of_work(factory, organization_id=current.organization_id) as db:
        org = await db.get(Organization, current.organization_id)
        record = await db.get(OrganizationDeletion, current.job_id)
        if org is None or org.deletion_requested_at is None or record is None:
            raise ValueError("Invalid deletion state")
    try:
        await asyncio.to_thread(
            write_marker, get_settings().deletion_journal_root, current.organization_id
        )
    except OSError as exc:
        raise RetryableJobError("Deletion journal is unavailable") from exc

    async def apply(db: AsyncSession, job: Job) -> None:
        await purge_tenant(db, current.organization_id, expected_job_id=job.id)

    return apply
