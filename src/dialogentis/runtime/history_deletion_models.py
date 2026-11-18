from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class HistoryDeletionTask(Base):
    __tablename__ = "history_deletion_tasks"

    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    request_key: Mapped[UUID | None] = mapped_column()
    filter_hash: Mapped[str] = mapped_column(String(64))
    selected_dialogues: Mapped[int] = mapped_column()
    deleted_dialogues: Mapped[int | None] = mapped_column()
    deleted_turns: Mapped[int | None] = mapped_column()
    cleared_receipts: Mapped[int | None] = mapped_column()

    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        UniqueConstraint("organization_id", "job_id"),
        UniqueConstraint("organization_id", "requested_by", "request_key"),
    )


class HistoryDeletionTarget(Base):
    __tablename__ = "history_deletion_targets"

    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    job_id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "job_id"],
            ["history_deletion_tasks.organization_id", "history_deletion_tasks.job_id"],
            ondelete="CASCADE",
        ),
    )
