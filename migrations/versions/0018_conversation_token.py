"""Hash opaque credentials for management test conversations."""

import sqlalchemy as sa
from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("test_conversations", sa.Column("credential_hash", sa.String(64), nullable=True))
    op.create_check_constraint(
        "ck_test_conversations_credential_hash",
        "test_conversations",
        "credential_hash IS NULL OR credential_hash ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_test_conversations_credential_hash", "test_conversations", type_="check")
    op.drop_column("test_conversations", "credential_hash")
