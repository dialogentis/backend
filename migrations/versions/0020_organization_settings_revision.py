"""Protect organization settings against concurrent changes."""

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("settings_revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_check_constraint(
        "ck_organizations_settings_revision", "organizations", "settings_revision > 0"
    )


def downgrade() -> None:
    op.drop_constraint("ck_organizations_settings_revision", "organizations", type_="check")
    op.drop_column("organizations", "settings_revision")
