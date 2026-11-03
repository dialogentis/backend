"""Track agent metadata revisions."""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agents", sa.Column("revision", sa.Integer(), nullable=False, server_default="1"))
    op.create_check_constraint("ck_agents_revision", "agents", "revision > 0")


def downgrade() -> None:
    op.drop_constraint("ck_agents_revision", "agents", type_="check")
    op.drop_column("agents", "revision")
