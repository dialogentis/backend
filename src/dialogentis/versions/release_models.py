from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from dialogentis.models import Base


class Release(Base):
    __tablename__ = "releases"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
    agent_id: Mapped[UUID] = mapped_column()
    root_version_id: Mapped[UUID] = mapped_column()
    dependency_manifest: Mapped[list[dict]] = mapped_column(JSONB)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reason: Mapped[str | None] = mapped_column(String(500))

    __table_args__ = (
        UniqueConstraint("organization_id", "agent_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "agent_id", "root_version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
        ),
        Index("ix_releases_agent_created", "organization_id", "agent_id", "created_at", "id"),
    )


class ReleaseVersion(Base):
    __tablename__ = "release_versions"

    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    release_agent_id: Mapped[UUID] = mapped_column()
    release_id: Mapped[UUID] = mapped_column(primary_key=True)
    agent_id: Mapped[UUID] = mapped_column()
    version_id: Mapped[UUID] = mapped_column(primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "release_agent_id", "release_id"],
            ["releases.organization_id", "releases.agent_id", "releases.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["organization_id", "agent_id", "version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
        ),
        Index("ix_release_versions_agent", "organization_id", "agent_id"),
    )
