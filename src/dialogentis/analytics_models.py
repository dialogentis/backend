from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class MetricEvent(Base):
    __tablename__ = "metric_events"

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    event_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    bucket_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_metric_events_expiry", "organization_id", "bucket_at"),)


class MetricHour(Base):
    __tablename__ = "metric_hours"

    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    bucket_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    root_agent_id: Mapped[UUID] = mapped_column(primary_key=True)
    active_agent_id: Mapped[UUID] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String(8), primary_key=True)
    metric: Mapped[str] = mapped_column(String(24), primary_key=True)
    item: Mapped[str] = mapped_column(String(36), primary_key=True)
    count: Mapped[int] = mapped_column(BigInteger)
    minute_counts: Mapped[dict[str, int]] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint("channel IN ('test', 'api', 'widget')"),
        CheckConstraint("count > 0"),
    )
