"""Attach published conversations to a website widget."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

PREVIOUS = (
    "(created_by IS NOT NULL AND api_key_id IS NULL AND release_id IS NULL) OR "
    "(created_by IS NULL AND api_key_id IS NOT NULL AND release_id IS NOT NULL "
    "AND http_mode = 'live')"
)


def upgrade() -> None:
    op.add_column("test_conversations", sa.Column("widget_id", pg.UUID))
    op.create_foreign_key(
        "fk_conversations_widget_id",
        "test_conversations",
        "widget_channels",
        ["organization_id", "agent_id", "widget_id"],
        ["organization_id", "agent_id", "id"],
    )
    op.drop_constraint("ck_conversations_principal", "test_conversations", type_="check")
    op.create_check_constraint(
        "ck_conversations_principal",
        "test_conversations",
        "(created_by IS NOT NULL AND api_key_id IS NULL AND widget_id IS NULL "
        "AND release_id IS NULL) OR "
        "(created_by IS NULL AND num_nonnulls(api_key_id, widget_id) = 1 "
        "AND release_id IS NOT NULL AND http_mode = 'live')",
    )
    op.execute(
        "CREATE POLICY widget_lookup ON widget_channels FOR SELECT TO dialogentis_app "
        "USING (id = NULLIF(current_setting('app.presented_widget_id', true), '')::uuid)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY widget_lookup ON widget_channels")
    for table in ("runtime_receipts", "test_conversations"):
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
    op.execute(
        "DELETE FROM runtime_receipts WHERE conversation_id IN "
        "(SELECT id FROM test_conversations WHERE widget_id IS NOT NULL)"
    )
    op.execute("DELETE FROM test_conversations WHERE widget_id IS NOT NULL")
    for table in ("runtime_receipts", "test_conversations"):
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.drop_constraint("ck_conversations_principal", "test_conversations", type_="check")
    op.drop_constraint("fk_conversations_widget_id", "test_conversations", type_="foreignkey")
    op.drop_column("test_conversations", "widget_id")
    op.create_check_constraint("ck_conversations_principal", "test_conversations", PREVIOUS)
