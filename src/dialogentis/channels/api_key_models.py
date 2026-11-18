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

type KeyScope = Literal["conversation:create", "message:send"]


class AgentApiKey(Base):
    __tablename__ = "agent_api_keys"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column()
    agent_id: Mapped[UUID] = mapped_column()
    name: Mapped[str] = mapped_column(String(120))
    prefix: Mapped[str] = mapped_column(String(12))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    scopes: Mapped[list[KeyScope]] = mapped_column(ARRAY(String(32)))
    created_by: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("organization_id", "agent_id", "id"),
        ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("expires_at > created_at", name="ck_agent_api_keys_expiry"),
        CheckConstraint(
            "cardinality(scopes) BETWEEN 1 AND 2 AND "
            "scopes <@ ARRAY['conversation:create', 'message:send']::varchar[]",
            name="ck_agent_api_keys_scopes",
        ),
        Index("ix_agent_api_keys_agent_created", "organization_id", "agent_id", "created_at", "id"),
    )
