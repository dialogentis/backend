from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base

type HistoryFormat = Literal["json", "csv"]


class HistoryExportTask(Base):
    __tablename__ = "history_export_tasks"

    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    format: Mapped[HistoryFormat] = mapped_column(String(8))
    filters: Mapped[dict] = mapped_column(JSONB)
    settings_revision: Mapped[int] = mapped_column()
    artifact_id: Mapped[UUID | None] = mapped_column()
    download_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["organization_id", "artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        CheckConstraint("format IN ('json', 'csv')"),
        Index("ix_history_export_artifact", "organization_id", "artifact_id"),
    )
