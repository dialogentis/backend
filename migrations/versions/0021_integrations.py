"""Tenant-scoped metadata and encrypted integration credentials."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "integrations",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("base_url", sa.String(2048), nullable=False),
        sa.Column("methods", pg.JSONB, nullable=False),
        sa.Column("paths", pg.JSONB, nullable=False),
        sa.Column("response_fields", pg.JSONB, nullable=False),
        sa.Column("encrypted_secret", sa.LargeBinary),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("revision", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("organization_id", "id"),
        sa.UniqueConstraint("organization_id", "name"),
        sa.CheckConstraint("revision > 0"),
        sa.CheckConstraint(
            "jsonb_typeof(methods) = 'array' AND jsonb_array_length(methods) BETWEEN 1 AND 5"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(paths) = 'array' AND jsonb_array_length(paths) BETWEEN 1 AND 100"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(response_fields) = 'array' AND jsonb_array_length(response_fields) <= 100"
        ),
        sa.CheckConstraint(
            "encrypted_secret IS NULL OR (octet_length(encrypted_secret) BETWEEN 37 AND 2084 "
            "AND substring(encrypted_secret from 1 for 4) = decode('44474931', 'hex'))"
        ),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE integrations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE integrations FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON integrations USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("integrations")
