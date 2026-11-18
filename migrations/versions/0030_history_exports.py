"""Track temporary and permission-bound history exports."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "history_export_tasks",
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("requested_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("format", sa.String(8), nullable=False),
        sa.Column("filters", pg.JSONB, nullable=False),
        sa.Column("settings_revision", sa.Integer, nullable=False),
        sa.Column("artifact_id", pg.UUID),
        sa.Column("download_until", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        sa.CheckConstraint("format IN ('json', 'csv')"),
    )
    op.create_index(
        "ix_history_export_artifact", "history_export_tasks", ["organization_id", "artifact_id"]
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE history_export_tasks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE history_export_tasks FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON history_export_tasks "
        f"USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("history_export_tasks")
