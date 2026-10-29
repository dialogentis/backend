import asyncio
import json
import sys

from rasa.core.agent import Agent


async def run():
    model = Agent.load(sys.argv[1])
    for line in sys.stdin:
        try:
            data = json.loads(line)
            result = await model.parse_message(data["text"])
        except Exception as exc:  # noqa: BLE001 - Report worker errors over the protocol.
            result = {"error": str(exc)}
        print(
            "ATLAS_RESULT " + json.dumps(result, ensure_ascii=False, default=str),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(run())
