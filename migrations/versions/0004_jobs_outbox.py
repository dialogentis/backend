"""Tenant-scoped durable jobs and their transactional outbox."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False),
        sa.Column("kind", sa.String(80), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("cancel_requested", sa.Boolean, nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("organization_id", "id"),
        sa.CheckConstraint("status IN ('queued','running','succeeded','failed','cancelled')"),
        sa.CheckConstraint("attempts >= 0"),
    )
    op.create_index("ix_jobs_organization_id", "jobs", ["organization_id"])
    op.create_table(
        "job_outbox",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False),
        sa.Column("job_id", pg.UUID, nullable=False),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
    )
    op.create_index(
        "ix_job_outbox_pending",
        "job_outbox",
        ["available_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    for table in ("jobs", "job_outbox"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING (organization_id = {tenant}) "
            f"WITH CHECK (organization_id = {tenant})"
        )
    op.execute("GRANT USAGE ON SCHEMA public TO dialogentis_dispatcher")
    op.execute("GRANT SELECT ON job_outbox TO dialogentis_dispatcher")
    op.execute("GRANT UPDATE (published_at) ON job_outbox TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY dispatcher_outbox ON job_outbox TO dialogentis_dispatcher "
        "USING (true) WITH CHECK (true)"
    )


def downgrade() -> None:
    op.drop_table("job_outbox")
    op.drop_table("jobs")
