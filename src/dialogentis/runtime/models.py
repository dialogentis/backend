"""Encrypted state of an internal management test conversation."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    LargeBinary,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class TestConversation(Base):
    __tablename__ = "test_conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    agent_id: Mapped[UUID] = mapped_column()
    version_id: Mapped[UUID] = mapped_column()
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    checkpoint: Mapped[bytes | None] = mapped_column(LargeBinary)
    http_mode: Mapped[str] = mapped_column(String(4), default="mock", server_default="mock")
    credential_hash: Mapped[str | None] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(default=1)
    processing_token: Mapped[UUID | None] = mapped_column()
    processing_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    idle_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "agent_id", "version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("revision > 0"),
        CheckConstraint("http_mode IN ('mock', 'live')", name="ck_test_conversations_http_mode"),
        CheckConstraint(
            "credential_hash IS NULL OR credential_hash ~ '^[0-9a-f]{64}$'",
            name="ck_test_conversations_credential_hash",
        ),
        CheckConstraint(
            "(processing_token IS NULL AND processing_expires_at IS NULL) OR "
            "(processing_token IS NOT NULL AND processing_expires_at IS NOT NULL "
            "AND checkpoint IS NOT NULL)",
            name="ck_test_conversations_processing",
        ),
        CheckConstraint(
            "idle_expires_at <= absolute_expires_at AND "
            "absolute_expires_at <= created_at + interval '24 hours'"
        ),
        CheckConstraint(
            "checkpoint IS NULL OR (octet_length(checkpoint) BETWEEN 36 AND 2097188 "
            "AND substring(checkpoint from 1 for 4) = decode('44474331', 'hex'))"
        ),
        Index("ix_test_conversations_organization_expiry", "organization_id", "idle_expires_at"),
    )
