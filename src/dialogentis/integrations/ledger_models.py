"""Metadata-only ledger for external actions inside a reserved turn."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class ActionExecution(Base):
    __tablename__ = "action_executions"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    conversation_id: Mapped[UUID] = mapped_column()
    receipt_id: Mapped[UUID] = mapped_column()
    integration_id: Mapped[UUID] = mapped_column()
    node_id: Mapped[UUID] = mapped_column()
    processing_token: Mapped[UUID] = mapped_column()
    ordinal: Mapped[int] = mapped_column()
    method: Mapped[str] = mapped_column(String(6))
    status: Mapped[str] = mapped_column(String(24))
    http_status: Mapped[int | None] = mapped_column()
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "receipt_id", "conversation_id"],
            [
                "runtime_receipts.organization_id",
                "runtime_receipts.id",
                "runtime_receipts.conversation_id",
            ],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "integration_id"],
            ["integrations.organization_id", "integrations.id"],
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("organization_id", "receipt_id", "ordinal", name="uq_action_turn_ordinal"),
        CheckConstraint("ordinal BETWEEN 1 AND 3"),
        CheckConstraint("method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')"),
        CheckConstraint(
            "status IN ('started', 'succeeded', 'http_error', 'timeout', 'invalid_response', "
            "'unknown_result', 'network_error', 'address_blocked', 'request_not_allowed', "
            "'dns_error', 'invalid_request', 'invalid_credential')"
        ),
        CheckConstraint(
            "(status = 'started' AND finished_at IS NULL AND http_status IS NULL) OR "
            "(status <> 'started' AND finished_at IS NOT NULL)"
        ),
        CheckConstraint("http_status IS NULL OR http_status BETWEEN 100 AND 599"),
        CheckConstraint(
            "(status = 'succeeded' AND http_status IS NOT NULL "
            "AND http_status BETWEEN 200 AND 299) "
            "OR (status <> 'succeeded' AND http_status IS NULL)"
        ),
        Index("ix_action_executions_integration", "organization_id", "integration_id"),
        Index(
            "ix_action_executions_conversation_status",
            "organization_id",
            "conversation_id",
            "status",
        ),
    )
