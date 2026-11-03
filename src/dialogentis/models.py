from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    email_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    retention_days: Mapped[int] = mapped_column(default=30)
    store_message_text: Mapped[bool] = mapped_column(default=True)
    deletion_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (CheckConstraint("retention_days BETWEEN 1 AND 30"),)


class Membership(Base):
    __tablename__ = "memberships"
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))
    __table_args__ = (
        CheckConstraint("role IN ('owner','admin','editor','viewer')"),
        Index(
            "uq_organization_owner",
            "organization_id",
            unique=True,
            postgresql_where=text("role = 'owner'"),
        ),
    )


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(String(2000), default="")
    language: Mapped[str] = mapped_column(String(2), default="ru")
    status: Mapped[Literal["active", "archived"]] = mapped_column(String(16), default="active")
    revision: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    __table_args__ = (
        UniqueConstraint("organization_id", "id"),
        UniqueConstraint("organization_id", "slug"),
        CheckConstraint("language = 'ru'"),
        CheckConstraint("status IN ('active','archived')"),
        CheckConstraint("revision > 0"),
    )


class AgentDraft(Base):
    __tablename__ = "agent_drafts"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(index=True)
    agent_id: Mapped[UUID] = mapped_column(unique=True)
    revision: Mapped[int] = mapped_column(default=1)
    schema_version: Mapped[int] = mapped_column(default=1)
    configuration: Mapped[dict] = mapped_column(JSONB)
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        CheckConstraint("revision > 0 AND schema_version > 0"),
    )
