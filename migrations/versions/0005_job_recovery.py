"""Fencing and recovery state for durable jobs."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs", sa.Column("delivery_recoveries", sa.Integer, server_default="0", nullable=False)
    )
    op.alter_column("jobs", "delivery_recoveries", server_default=None)
    op.add_column("jobs", sa.Column("execution_token", pg.UUID))
    op.add_column("jobs", sa.Column("next_retry_at", sa.DateTime(timezone=True)))
    op.create_check_constraint(None, "jobs", "delivery_recoveries >= 0")
    op.create_index(
        "ix_job_outbox_job_status", "job_outbox", ["organization_id", "job_id", "published_at"]
    )
    op.create_table(
        "maintenance_runs",
        sa.Column("name", sa.String(80), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("maintenance_runs")
    op.drop_index("ix_job_outbox_job_status", table_name="job_outbox")
    op.drop_constraint("jobs_delivery_recoveries_check", "jobs", type_="check")
    op.drop_column("jobs", "next_retry_at")
    op.drop_column("jobs", "execution_token")
    op.drop_column("jobs", "delivery_recoveries")
