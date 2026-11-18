"""Queue confirmed history deletions with a fixed selection."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "history_deletion_tasks",
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("requested_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_key", pg.UUID),
        sa.Column("filter_hash", sa.String(64), nullable=False),
        sa.Column("selected_dialogues", sa.Integer, nullable=False),
        sa.Column("deleted_dialogues", sa.Integer),
        sa.Column("deleted_turns", sa.Integer),
        sa.Column("cleared_receipts", sa.Integer),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
        sa.UniqueConstraint("organization_id", "job_id"),
        sa.UniqueConstraint("organization_id", "requested_by", "request_key"),
    )
    op.create_table(
        "history_deletion_targets",
        sa.Column("organization_id", pg.UUID, primary_key=True),
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("conversation_id", pg.UUID, primary_key=True),
        sa.ForeignKeyConstraint(
            ["organization_id", "job_id"],
            ["history_deletion_tasks.organization_id", "history_deletion_tasks.job_id"],
            ondelete="CASCADE",
        ),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    for table in ("history_deletion_tasks", "history_deletion_targets"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING (organization_id = {tenant}) "
            f"WITH CHECK (organization_id = {tenant})"
        )


def downgrade() -> None:
    op.drop_table("history_deletion_targets")
    op.drop_table("history_deletion_tasks")
