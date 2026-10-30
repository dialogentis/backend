from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict] = Field(default_factory=list)
    request_id: UUID


class ErrorResponse(BaseModel):
    error: ErrorDetail


class CursorPage[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


def revision_etag(revision: int) -> str:
    if revision < 1:
        raise ValueError("Revision must be positive")
    return f'"rev-{revision}"'
