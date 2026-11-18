from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Index, LargeBinary, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dialogentis.models import Base


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    agent_id: Mapped[UUID] = mapped_column()
    version_id: Mapped[UUID] = mapped_column()
    release_id: Mapped[UUID | None] = mapped_column()
    channel_type: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("channel_type IN ('test','api','widget')"),
        CheckConstraint("status IN ('waiting_input','completed','failed','expired')"),
        Index("ix_history_created", "organization_id", "created_at", "id"),
        Index("ix_history_activity", "organization_id", "last_activity_at"),
    )


class HistoryTurn(Base):
    __tablename__ = "conversation_history_turns"

    organization_id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(primary_key=True)
    sequence: Mapped[int] = mapped_column(primary_key=True)
    turn_id: Mapped[UUID] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    intent_id: Mapped[UUID | None] = mapped_column()
    error_code: Mapped[str | None] = mapped_column(String(64))
    trace: Mapped[list[dict]] = mapped_column(JSONB)
    encrypted_messages: Mapped[bytes | None] = mapped_column(LargeBinary)

    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "conversation_id"],
            ["conversation_history.organization_id", "conversation_history.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("sequence >= 0"),
        Index("ix_history_turn_created", "organization_id", "created_at"),
    )
