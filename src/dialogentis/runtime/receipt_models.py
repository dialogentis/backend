"""Tenant-isolated message receipts; payloads, when present, are encrypted."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class RuntimeReceipt(Base):
    __tablename__ = "runtime_receipts"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    conversation_id: Mapped[UUID] = mapped_column()
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    credential_hash: Mapped[str] = mapped_column(String(64))
    client_message_id: Mapped[UUID] = mapped_column()
    idempotency_key_hash: Mapped[str] = mapped_column(String(64))
    request_digest: Mapped[str] = mapped_column(String(64))
    key_version: Mapped[int] = mapped_column(BigInteger)
    processing_token: Mapped[UUID] = mapped_column()
    status: Mapped[str] = mapped_column(String(16))
    revision: Mapped[int | None] = mapped_column()
    runtime_status: Mapped[str | None] = mapped_column(String(16))
    encrypted_response: Mapped[bytes | None] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "organization_id", "conversation_id", "client_message_id", name="uq_receipt_message"
        ),
        UniqueConstraint(
            "organization_id", "conversation_id", "idempotency_key_hash", name="uq_receipt_key"
        ),
        CheckConstraint(
            "credential_hash ~ '^[0-9a-f]{64}$' AND idempotency_key_hash ~ '^[0-9a-f]{64}$' "
            "AND request_digest ~ '^[0-9a-f]{64}$'"
        ),
        CheckConstraint("key_version > 0 AND key_version <= 4294967295"),
        CheckConstraint("status IN ('processing', 'completed', 'unknown')"),
        CheckConstraint(
            "(status = 'completed' AND revision IS NOT NULL AND revision > 0 "
            "AND runtime_status IS NOT NULL "
            "AND runtime_status IN ('waiting_input', 'completed', 'failed')) "
            "OR (status <> 'completed' AND revision IS NULL AND runtime_status IS NULL "
            "AND encrypted_response IS NULL)"
        ),
        CheckConstraint(
            "encrypted_response IS NULL OR "
            "(octet_length(encrypted_response) BETWEEN 36 AND 2097188 "
            "AND substring(encrypted_response from 1 for 4) = decode('44475231', 'hex'))"
        ),
        Index("ix_runtime_receipts_org_expiry", "organization_id", "expires_at"),
    )
