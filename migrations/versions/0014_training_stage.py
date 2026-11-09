"""Expose actual training stages without inventing percentage progress."""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("stage", sa.String(32)))
    op.create_check_constraint(
        "ck_jobs_stage",
        "jobs",
        "stage IS NULL OR stage IN ('validation','intent_training','entity_training',"
        "'evaluation','packaging')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_jobs_stage", "jobs", type_="check")
    op.drop_column("jobs", "stage")
