"""Tenant-scoped import previews and their verified payloads."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "import_tasks",
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("requested_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("mode", sa.String(8), nullable=False),
        sa.Column("base_revision", sa.Integer, nullable=False),
        sa.Column("input_artifact_id", pg.UUID),
        sa.Column("candidate", pg.JSONB),
        sa.Column("report", pg.JSONB),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("applied_revision", sa.Integer),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "input_artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        sa.CheckConstraint("format IN ('json', 'csv')"),
        sa.CheckConstraint("mode IN ('merge', 'replace')"),
        sa.CheckConstraint(
            "base_revision > 0 AND (applied_revision IS NULL OR applied_revision > base_revision)"
        ),
    )
    op.create_index("ix_import_tasks_organization_id", "import_tasks", ["organization_id"])
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE import_tasks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE import_tasks FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON import_tasks USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("import_tasks")
