from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base

type WidgetPosition = Literal["bottom-left", "bottom-right"]


class WidgetChannel(Base):
    __tablename__ = "widget_channels"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column()
    agent_id: Mapped[UUID] = mapped_column()
    name: Mapped[str] = mapped_column(String(120))
    welcome_text: Mapped[str] = mapped_column(String(1000))
    accent_color: Mapped[str] = mapped_column(String(7))
    position: Mapped[WidgetPosition] = mapped_column(String(16))
    allowed_origins: Mapped[list[str]] = mapped_column(ARRAY(String(512)))
    enabled: Mapped[bool] = mapped_column()
    revision: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))

    __table_args__ = (
        UniqueConstraint("organization_id", "agent_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "revision > 0 AND cardinality(allowed_origins) BETWEEN 1 AND 20 "
            "AND accent_color ~ '^#[a-f0-9]{6}$' "
            "AND position IN ('bottom-left', 'bottom-right')",
            name="ck_widget_channels_settings",
        ),
        Index(
            "ix_widget_channels_agent_created", "organization_id", "agent_id", "created_at", "id"
        ),
    )
