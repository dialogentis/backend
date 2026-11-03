from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse


class DomainValidationError(Exception):
    def __init__(self, details: list[dict]):
        self.details = details


def error_response(request: Request, status: int, code: str, message: str, details=None):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
                "request_id": getattr(request.state, "request_id", str(uuid4())),
            }
        },
    )
