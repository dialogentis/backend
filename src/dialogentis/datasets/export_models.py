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


class ExportTask(Base):
    __tablename__ = "export_tasks"

    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    agent_id: Mapped[UUID] = mapped_column()
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    format: Mapped[Literal["json", "csv"]] = mapped_column(String(8))
    revision: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    snapshot_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    artifact_id: Mapped[UUID | None] = mapped_column()
    download_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        ForeignKeyConstraint(
            ["organization_id", "artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        CheckConstraint("format IN ('json', 'csv')"),
        CheckConstraint("revision > 0"),
        Index("ix_export_tasks_organization_id", "organization_id"),
    )
