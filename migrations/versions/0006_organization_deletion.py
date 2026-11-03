"""Durable organization-deletion requests and tenant-tagged mail."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("organizations", sa.Column("deletion_requested_at", sa.DateTime(timezone=True)))
    op.add_column("email_outbox", sa.Column("organization_id", pg.UUID))
    op.create_index("ix_email_outbox_organization_id", "email_outbox", ["organization_id"])
    op.create_table(
        "organization_deletions",
        sa.Column("job_id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, nullable=False, unique=True),
        sa.Column("requested_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["organization_id", "job_id"], ["jobs.organization_id", "jobs.id"]),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    user = "NULLIF(current_setting('app.user_id', true), '')::uuid"
    op.execute("ALTER TABLE organization_deletions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE organization_deletions FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY deletion_read ON organization_deletions FOR SELECT "
        f"USING (organization_id = {tenant} OR requested_by = {user})"
    )
    op.execute(
        "CREATE POLICY deletion_insert ON organization_deletions FOR INSERT "
        f"WITH CHECK (organization_id = {tenant} AND requested_by = {user})"
    )
    op.execute(
        "CREATE POLICY deletion_update ON organization_deletions FOR UPDATE "
        f"USING (organization_id = {tenant}) WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("organization_deletions")
    op.drop_index("ix_email_outbox_organization_id", table_name="email_outbox")
    op.drop_column("email_outbox", "organization_id")
    op.drop_column("organizations", "deletion_requested_at")
