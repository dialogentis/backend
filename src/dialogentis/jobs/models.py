from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base

JOB_QUEUE_NAMES = {"normal": "dialogentis.jobs", "training": "dialogentis.training"}


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Job metadata must survive removal of the organization it belongs to.
    organization_id: Mapped[UUID] = mapped_column(index=True)
    kind: Mapped[str] = mapped_column(String(80))
    queue: Mapped[Literal["normal", "training"]] = mapped_column(
        String(16), default="normal", server_default=text("'normal'")
    )
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    delivery_recoveries: Mapped[int] = mapped_column(default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    stage: Mapped[str | None] = mapped_column(String(32))
    execution_token: Mapped[UUID | None] = mapped_column()
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("organization_id", "id", "queue", name="uq_jobs_org_id_queue"),
        CheckConstraint(
            "(kind = 'agent.training' AND queue = 'training') OR "
            "(kind <> 'agent.training' AND queue = 'normal')",
            name="ck_jobs_kind_queue",
        ),
        CheckConstraint("status IN ('queued','running','succeeded','failed','cancelled')"),
        CheckConstraint("attempts >= 0"),
        CheckConstraint("delivery_recoveries >= 0"),
        CheckConstraint(
            "stage IS NULL OR stage IN ('validation','intent_training','entity_training',"
            "'evaluation','packaging')",
            name="ck_jobs_stage",
        ),
    )


class OutboxEvent(Base):
    __tablename__ = "job_outbox"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column()
    job_id: Mapped[UUID] = mapped_column()
    queue: Mapped[Literal["normal", "training"]] = mapped_column(
        String(16), default="normal", server_default=text("'normal'")
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        ForeignKeyConstraint(
            ["organization_id", "job_id", "queue"],
            ["jobs.organization_id", "jobs.id", "jobs.queue"],
            name="fk_outbox_job_queue",
        ),
        Index("ix_job_outbox_pending", "available_at", postgresql_where=published_at.is_(None)),
        Index("ix_job_outbox_job_status", "organization_id", "job_id", "published_at"),
    )


class MaintenanceRun(Base):
    __tablename__ = "maintenance_runs"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
