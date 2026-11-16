"""Fenced reservation for one in-progress management turn."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("test_conversations", sa.Column("processing_token", pg.UUID, nullable=True))
    op.add_column(
        "test_conversations",
        sa.Column("processing_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_test_conversations_processing",
        "test_conversations",
        "(processing_token IS NULL AND processing_expires_at IS NULL) OR "
        "(processing_token IS NOT NULL AND processing_expires_at IS NOT NULL "
        "AND checkpoint IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_test_conversations_processing", "test_conversations", type_="check")
    op.drop_column("test_conversations", "processing_expires_at")
    op.drop_column("test_conversations", "processing_token")
