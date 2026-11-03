"""Exclusive, fenced editing leases on agent drafts."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_drafts", sa.Column("lease_holder_id", pg.UUID, sa.ForeignKey("users.id")))
    op.add_column("agent_drafts", sa.Column("lease_tab_id", pg.UUID))
    op.add_column("agent_drafts", sa.Column("lease_token_hash", sa.String(64)))
    op.add_column(
        "agent_drafts", sa.Column("lease_fence", sa.BigInteger, nullable=False, server_default="0")
    )
    op.add_column("agent_drafts", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
    op.create_check_constraint("ck_lease_fence", "agent_drafts", "lease_fence >= 0")
    op.create_check_constraint(
        "ck_lease_fields",
        "agent_drafts",
        "(lease_holder_id IS NULL AND lease_tab_id IS NULL AND lease_token_hash IS NULL "
        "AND lease_expires_at IS NULL) OR (lease_holder_id IS NOT NULL "
        "AND lease_tab_id IS NOT NULL "
        "AND lease_token_hash IS NOT NULL AND lease_expires_at IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_lease_fields", "agent_drafts", type_="check")
    op.drop_constraint("ck_lease_fence", "agent_drafts", type_="check")
    for column in (
        "lease_expires_at",
        "lease_fence",
        "lease_token_hash",
        "lease_tab_id",
        "lease_holder_id",
    ):
        op.drop_column("agent_drafts", column)
