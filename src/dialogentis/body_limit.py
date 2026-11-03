from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dialogentis.api_errors import error_response


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_bytes:
                    response = error_response(
                        Request(scope), 413, "payload_too_large", "Request body is too large"
                    )
                    await response(scope, receive, send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
