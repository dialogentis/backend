"""Tenant-scoped credentials for public agent conversations."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_api_keys",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("prefix", sa.String(12), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("scopes", pg.ARRAY(sa.String(32)), nullable=False),
        sa.Column("created_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("organization_id", "agent_id", "id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("expires_at > created_at", name="ck_agent_api_keys_expiry"),
        sa.CheckConstraint(
            "cardinality(scopes) BETWEEN 1 AND 2 AND "
            "scopes <@ ARRAY['conversation:create', 'message:send']::varchar[]",
            name="ck_agent_api_keys_scopes",
        ),
    )
    op.create_index(
        "ix_agent_api_keys_agent_created",
        "agent_api_keys",
        ["organization_id", "agent_id", "created_at", "id"],
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE agent_api_keys ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_api_keys FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON agent_api_keys USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )
    op.execute("REVOKE UPDATE ON agent_api_keys FROM dialogentis_app")
    op.execute("GRANT UPDATE (last_used_at, revoked_at) ON agent_api_keys TO dialogentis_app")


def downgrade() -> None:
    op.drop_table("agent_api_keys")
