from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class Integration(Base):
    __tablename__ = "integrations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(120))
    base_url: Mapped[str] = mapped_column(String(2048))
    methods: Mapped[list[str]] = mapped_column(JSONB)
    paths: Mapped[list[str]] = mapped_column(JSONB)
    response_fields: Mapped[list[str]] = mapped_column(JSONB)
    encrypted_secret: Mapped[bytes | None] = mapped_column(LargeBinary)
    enabled: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    revision: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("organization_id", "name"),
        CheckConstraint("revision > 0"),
        CheckConstraint(
            "jsonb_typeof(methods) = 'array' AND jsonb_array_length(methods) BETWEEN 1 AND 5"
        ),
        CheckConstraint(
            "jsonb_typeof(paths) = 'array' AND jsonb_array_length(paths) BETWEEN 1 AND 100"
        ),
        CheckConstraint(
            "jsonb_typeof(response_fields) = 'array' AND jsonb_array_length(response_fields) <= 100"
        ),
        CheckConstraint(
            "encrypted_secret IS NULL OR (octet_length(encrypted_secret) BETWEEN 37 AND 2084 "
            "AND substring(encrypted_secret from 1 for 4) = decode('44474931', 'hex'))"
        ),
    )
