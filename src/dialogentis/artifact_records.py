from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
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


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    job_id: Mapped[UUID | None] = mapped_column()
    kind: Mapped[Literal["temporary", "model"]] = mapped_column(String(16))
    status: Mapped[Literal["reserved", "ready"]] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        CheckConstraint("kind IN ('temporary', 'model')"),
        CheckConstraint("status IN ('reserved', 'ready')"),
        CheckConstraint("size_bytes >= 0"),
        CheckConstraint(
            "(status = 'reserved' AND sha256 IS NULL AND expires_at IS NOT NULL) OR "
            "(status = 'ready' AND sha256 IS NOT NULL AND sha256 ~ '^[0-9a-f]{64}$' AND "
            "((kind = 'temporary' AND expires_at IS NOT NULL) OR "
            "(kind = 'model' AND expires_at IS NULL)))"
        ),
        Index("ix_artifacts_org_kind", "organization_id", "kind"),
        Index("ix_artifacts_org_expiry", "organization_id", "expires_at"),
    )
