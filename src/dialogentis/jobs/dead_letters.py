import logging
from uuid import UUID

from aio_pika.exceptions import ChannelNotFoundEntity
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dialogentis.jobs.broker import JobBroker
from dialogentis.jobs.lifecycle import recover_dead_letter
from dialogentis.jobs.models import JOB_QUEUE_NAMES
from dialogentis.jobs.tasks import run_job
from dialogentis.jobs.training_tasks import run_training_job


async def recover_dead_letters(
    factory: async_sessionmaker[AsyncSession], broker: JobBroker
) -> None:
    connection = broker.write_conn
    if connection is None:
        return
    for queue, task in (("normal", run_job), ("training", run_training_job)):
        name = JOB_QUEUE_NAMES[queue]
        async with connection.channel() as channel:
            try:
                dead = await channel.get_queue(f"{name}.dead", ensure=True)
            except ChannelNotFoundEntity:
                continue
            for _ in range(20):
                delivery = await dead.get(fail=False, timeout=5)
                if delivery is None:
                    break
                async with delivery.process(requeue=True):
                    try:
                        message = broker.formatter.loads(delivery.body)
                        if (
                            delivery.headers.get("x-last-death-queue") != name
                            or message.task_name != task.task_name
                            or message.kwargs
                        ):
                            raise ValueError("Not a dispatched job")
                        job_id, organization_id = (UUID(str(value)) for value in message.args)
                        event_id = UUID(message.task_id)
                    except TypeError, ValueError:
                        logging.warning("Ignoring an unrecognized dead-letter delivery")
                        continue
                    await recover_dead_letter(factory, organization_id, job_id, event_id, queue)
