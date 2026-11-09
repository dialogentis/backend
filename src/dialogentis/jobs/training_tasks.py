"""Dedicated training queue; handlers are registered with the training worker."""

from uuid import UUID

from taskiq import Context, TaskiqDepends

from dialogentis.jobs.broker import create_broker
from dialogentis.jobs.tasks import Handler, _run
from dialogentis.training.worker import prepare_training

broker = create_broker("dialogentis.training")
HANDLERS: dict[str, Handler] = {
    "agent.training": Handler(prepare_training, timeout_seconds=900, retryable=False)
}


@broker.task(ack_type="manual")
async def run_training_job(
    job_id: UUID, organization_id: UUID, context: Context = TaskiqDepends()
) -> None:
    await _run(broker, job_id, organization_id, context, HANDLERS, "training")
