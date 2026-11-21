from time import monotonic
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.jobs.models import Job, OutboxEvent
from dialogentis.telemetry import log_event, request_id


def enqueue(db: AsyncSession, organization_id: UUID, kind: str) -> Job:
    """Caller verifies membership and tenant existence, then owns the transaction."""
    started = monotonic()
    queue = "training" if kind == "agent.training" else "normal"
    job = Job(id=uuid4(), organization_id=organization_id, kind=kind, queue=queue)
    db.add(job)
    db.add(OutboxEvent(organization_id=organization_id, job_id=job.id, queue=queue))
    if request_id.get() is not None:
        log_event("api", "job_enqueue_requested", started=started, job_id=job.id)
    return job
