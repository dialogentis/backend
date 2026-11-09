"""Durable, tenant-scoped configuration export requests."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "export_tasks",
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("requested_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("snapshot", pg.JSONB),
        sa.Column("snapshot_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_id", pg.UUID),
        sa.Column("download_until", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        sa.CheckConstraint("format IN ('json', 'csv')"),
        sa.CheckConstraint("revision > 0"),
    )
    op.create_index("ix_export_tasks_organization_id", "export_tasks", ["organization_id"])
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE export_tasks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE export_tasks FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON export_tasks USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("export_tasks")
