"""Tenant-isolated encrypted test-conversation state."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "test_conversations",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("version_id", pg.UUID, nullable=False),
        sa.Column("created_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("checkpoint", sa.LargeBinary),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("idle_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("absolute_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id", "version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("revision > 0"),
        sa.CheckConstraint(
            "idle_expires_at <= absolute_expires_at AND "
            "absolute_expires_at <= created_at + interval '24 hours'"
        ),
        sa.CheckConstraint(
            "checkpoint IS NULL OR (octet_length(checkpoint) BETWEEN 36 AND 2097188 "
            "AND substring(checkpoint from 1 for 4) = decode('44474331', 'hex'))"
        ),
    )
    op.create_index(
        "ix_test_conversations_organization_expiry",
        "test_conversations",
        ["organization_id", "idle_expires_at"],
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE test_conversations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE test_conversations FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON test_conversations USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("test_conversations")
