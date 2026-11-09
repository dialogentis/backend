"""Bind durable outbox events to their job's dedicated queue."""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("jobs", "job_outbox"):
        op.add_column(
            table, sa.Column("queue", sa.String(16), nullable=False, server_default="normal")
        )
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
    op.execute("UPDATE jobs SET queue = 'training' WHERE kind = 'agent.training'")
    op.execute(
        "UPDATE job_outbox AS o SET queue = j.queue FROM jobs AS j "
        "WHERE o.organization_id = j.organization_id AND o.job_id = j.id"
    )
    for table in ("jobs", "job_outbox"):
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.create_unique_constraint("uq_jobs_org_id_queue", "jobs", ["organization_id", "id", "queue"])
    op.create_check_constraint(
        "ck_jobs_kind_queue",
        "jobs",
        "(kind = 'agent.training' AND queue = 'training') OR "
        "(kind <> 'agent.training' AND queue = 'normal')",
    )
    op.create_foreign_key(
        "fk_outbox_job_queue",
        "job_outbox",
        "jobs",
        ["organization_id", "job_id", "queue"],
        ["organization_id", "id", "queue"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_outbox_job_queue", "job_outbox", type_="foreignkey")
    op.drop_constraint("ck_jobs_kind_queue", "jobs", type_="check")
    op.drop_constraint("uq_jobs_org_id_queue", "jobs", type_="unique")
    op.drop_column("job_outbox", "queue")
    op.drop_column("jobs", "queue")
