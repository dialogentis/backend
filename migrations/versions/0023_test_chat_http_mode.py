"""Persist the HTTP mode of a test conversation."""

import sqlalchemy as sa
from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "integrations",
        sa.Column("supports_idempotency", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "test_conversations",
        sa.Column("http_mode", sa.String(4), nullable=False, server_default="mock"),
    )
    op.create_check_constraint(
        "ck_test_conversations_http_mode", "test_conversations", "http_mode IN ('mock', 'live')"
    )


def downgrade() -> None:
    op.drop_constraint("ck_test_conversations_http_mode", "test_conversations", type_="check")
    op.drop_column("test_conversations", "http_mode")
    op.drop_column("integrations", "supports_idempotency")
