import inspect
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

from taskiq import AckableMessage
from taskiq_aio_pika import AioPikaBroker, Exchange, Queue
from taskiq_aio_pika.exceptions import NoStartupError

from dialogentis.config import get_settings


class JobBroker(AioPikaBroker):
    async def listen(self) -> AsyncGenerator[AckableMessage]:
        if self.read_channel is None:
            raise NoStartupError("Call startup before listening")
        await self.read_channel.set_qos(prefetch_count=1)
        queue = await self.read_channel.get_queue(self._task_queues[0].name, ensure=True)
        async with queue.iterator() as iterator:
            async for message in iterator:
                try:
                    task_message = self.formatter.loads(message.body)
                    task_message.parse_labels()
                    task = self.find_task(task_message.task_name)
                    if task is None or task_message.labels.get("ack_type") != "manual":
                        raise ValueError("Unknown task or invalid acknowledgement mode")
                    signature = inspect.signature(task.original_func)
                    arguments = signature.bind(*task_message.args, **task_message.kwargs)
                    if "context" in arguments.arguments:
                        raise ValueError("Task context cannot be supplied in a message")
                    for name, value in arguments.arguments.items():
                        if signature.parameters[name].annotation is UUID:
                            UUID(str(value))
                except Exception:
                    logging.warning("Rejecting malformed job message")
                    await message.reject(requeue=False)
                    continue
                yield AckableMessage(data=message.body, ack=message.ack)


def create_broker(queue_name: str = "dialogentis.jobs") -> JobBroker:
    url = get_settings().broker_url
    if url is None:
        raise ValueError("BROKER_URL is required for workers")
    return JobBroker(
        url.get_secret_value(),
        qos=1,
        exchange=Exchange(name="dialogentis", declare=True, durable=True),
        task_queues=[Queue(name=queue_name, durable=True, arguments={"x-delivery-limit": 3})],
        dead_letter_queue=Queue(name=f"{queue_name}.dead", durable=True),
    )


broker = create_broker()
