"""Save per-agent widget configuration and its exact browser origins."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "widget_channels",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("welcome_text", sa.String(1000), nullable=False),
        sa.Column("accent_color", sa.String(7), nullable=False),
        sa.Column("position", sa.String(16), nullable=False),
        sa.Column("allowed_origins", pg.ARRAY(sa.String(512)), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.UniqueConstraint("organization_id", "agent_id", "id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "revision > 0 AND cardinality(allowed_origins) BETWEEN 1 AND 20 "
            "AND accent_color ~ '^#[a-f0-9]{6}$' "
            "AND position IN ('bottom-left', 'bottom-right')",
            name="ck_widget_channels_settings",
        ),
    )
    op.create_index(
        "ix_widget_channels_agent_created",
        "widget_channels",
        ["organization_id", "agent_id", "created_at", "id"],
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE widget_channels ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE widget_channels FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON widget_channels USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("widget_channels")
