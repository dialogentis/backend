from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.jobs.models import Job, OutboxEvent


def enqueue(db: AsyncSession, organization_id: UUID, kind: str) -> Job:
    """Caller verifies membership and tenant existence, then owns the transaction."""
    job = Job(id=uuid4(), organization_id=organization_id, kind=kind)
    db.add(job)
    db.add(OutboxEvent(organization_id=organization_id, job_id=job.id))
    return job
