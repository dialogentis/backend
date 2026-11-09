from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class AgentVersion(Base):
    __tablename__ = "agent_versions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    agent_id: Mapped[UUID] = mapped_column()
    number: Mapped[int] = mapped_column()
    source_revision: Mapped[int] = mapped_column()
    snapshot: Mapped[dict] = mapped_column(JSONB)
    config_hash: Mapped[str] = mapped_column(String(64))
    nlu_hash: Mapped[str] = mapped_column(String(64))
    dataset_warnings: Mapped[list] = mapped_column(JSONB)
    warnings_truncated: Mapped[bool] = mapped_column(default=False)
    job_id: Mapped[UUID] = mapped_column(unique=True)
    queue: Mapped[str] = mapped_column(
        String(16), server_default=text("'training'"), default="training"
    )
    idempotency_key_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_artifact_id: Mapped[UUID | None] = mapped_column()
    validation_report: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))

    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("organization_id", "agent_id", "id"),
        UniqueConstraint("organization_id", "agent_id", "number"),
        UniqueConstraint("organization_id", "agent_id", "created_by", "idempotency_key_hash"),
        Index("ix_agent_versions_nlu_hash", "organization_id", "agent_id", "nlu_hash"),
        ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        ForeignKeyConstraint(
            ["organization_id", "job_id", "queue"],
            ["jobs.organization_id", "jobs.id", "jobs.queue"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "model_artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        CheckConstraint("number > 0 AND source_revision > 0"),
        CheckConstraint("queue = 'training'"),
        CheckConstraint(
            "config_hash ~ '^[0-9a-f]{64}$' AND nlu_hash ~ '^[0-9a-f]{64}$' "
            "AND idempotency_key_hash ~ '^[0-9a-f]{64}$'"
        ),
        CheckConstraint("(model_artifact_id IS NULL) = (validation_report IS NULL)"),
    )
