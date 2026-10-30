"""Only loaded by the isolated integration-test worker, never by the API."""

import os
from pathlib import Path
from uuid import UUID

from taskiq import Context, TaskiqDepends

from dialogentis.jobs.broker import create_broker

broker = create_broker("dialogentis.smoke")


@broker.task(ack_type="manual")
async def probe(nonce: UUID, context: Context = TaskiqDepends()) -> None:
    root = Path(os.environ["TEST_OUTPUT_DIR"])
    with (root / str(nonce)).open("w") as stream:
        stream.write("processed")
        stream.flush()
        os.fsync(stream.fileno())
    await context.ack()
