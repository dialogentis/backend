from taskiq_aio_pika import AioPikaBroker, Exchange, Queue

from dialogentis.config import get_settings


def create_broker(queue_name: str = "dialogentis.jobs") -> AioPikaBroker:
    url = get_settings().broker_url
    if url is None:
        raise ValueError("BROKER_URL is required for workers")
    return AioPikaBroker(
        url.get_secret_value(),
        qos=1,
        exchange=Exchange(name="dialogentis", declare=True, durable=True),
        task_queues=[Queue(name=queue_name, durable=True)],
        dead_letter_queue=Queue(name=f"{queue_name}.dead", durable=True),
    )


broker = create_broker()
