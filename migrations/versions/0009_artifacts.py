"""Tenant-scoped artifact reservations and immutable file metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("job_id", pg.UUID),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("sha256", sa.String(64)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("organization_id", "id"),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        sa.CheckConstraint("kind IN ('temporary', 'model')"),
        sa.CheckConstraint("status IN ('reserved', 'ready')"),
        sa.CheckConstraint("size_bytes >= 0"),
        sa.CheckConstraint(
            "(status = 'reserved' AND sha256 IS NULL AND expires_at IS NOT NULL) OR "
            "(status = 'ready' AND sha256 IS NOT NULL AND sha256 ~ '^[0-9a-f]{64}$' AND "
            "((kind = 'temporary' AND expires_at IS NOT NULL) OR "
            "(kind = 'model' AND expires_at IS NULL)))"
        ),
    )
    op.create_index("ix_artifacts_org_kind", "artifacts", ["organization_id", "kind"])
    op.create_index("ix_artifacts_org_expiry", "artifacts", ["organization_id", "expires_at"])
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE artifacts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE artifacts FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON artifacts USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )
    op.execute("GRANT SELECT (organization_id) ON artifacts TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY dispatcher_artifacts ON artifacts FOR SELECT "
        "TO dialogentis_dispatcher USING (true)"
    )


def downgrade() -> None:
    op.drop_table("artifacts")
