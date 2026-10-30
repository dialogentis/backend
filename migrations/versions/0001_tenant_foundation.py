"""Users, organizations and tenant boundaries."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("email_verified", sa.Boolean, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "organizations",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("retention_days", sa.Integer, nullable=False),
        sa.Column("store_message_text", sa.Boolean, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("retention_days BETWEEN 1 AND 30"),
    )
    op.create_table(
        "memberships",
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), primary_key=True),
        sa.Column("user_id", pg.UUID, sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.CheckConstraint("role IN ('owner','admin','editor','viewer')"),
    )
    op.create_table(
        "agents",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("description", sa.String(2000), nullable=False),
        sa.Column("language", sa.String(2), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.UniqueConstraint("organization_id", "id"),
        sa.UniqueConstraint("organization_id", "slug"),
        sa.CheckConstraint("language = 'ru'"),
        sa.CheckConstraint("status IN ('active','archived')"),
    )
    op.create_index("ix_agents_organization_id", "agents", ["organization_id"])
    op.create_table(
        "agent_drafts",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False, unique=True),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.Column("configuration", pg.JSONB, nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        sa.CheckConstraint("revision > 0 AND schema_version > 0"),
    )
    op.create_index("ix_agent_drafts_organization_id", "agent_drafts", ["organization_id"])
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    user = "NULLIF(current_setting('app.user_id', true), '')::uuid"
    for table, expression in {
        "organizations": f"id = {tenant}",
        "memberships": f"organization_id = {tenant} OR user_id = {user}",
        "agents": f"organization_id = {tenant}",
        "agent_drafts": f"organization_id = {tenant}",
    }.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({expression}) WITH CHECK ({expression})"
        )


def downgrade() -> None:
    for table in ["agent_drafts", "agents", "memberships", "organizations", "users"]:
        op.drop_table(table)
