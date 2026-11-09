from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class ImportTask(Base):
    __tablename__ = "import_tasks"

    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    agent_id: Mapped[UUID] = mapped_column()
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    format: Mapped[Literal["json", "csv"]] = mapped_column(String(8))
    mode: Mapped[Literal["merge", "replace"]] = mapped_column(String(8))
    base_revision: Mapped[int] = mapped_column(Integer)
    input_artifact_id: Mapped[UUID | None] = mapped_column()
    candidate: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    report: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    applied_revision: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        ForeignKeyConstraint(
            ["organization_id", "input_artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        CheckConstraint("format IN ('json', 'csv')"),
        CheckConstraint("mode IN ('merge', 'replace')"),
        CheckConstraint(
            "base_revision > 0 AND (applied_revision IS NULL OR applied_revision > base_revision)"
        ),
        Index("ix_import_tasks_organization_id", "organization_id"),
    )
