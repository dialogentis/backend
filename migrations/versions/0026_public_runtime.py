"""Bind public conversations to their API key and immutable release."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_rate_buckets",
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("subject", sa.String(64), primary_key=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE runtime_rate_buckets ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE runtime_rate_buckets FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY tenant_isolation ON runtime_rate_buckets "
        f"USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )
    op.add_column("test_conversations", sa.Column("api_key_id", pg.UUID))
    op.add_column("test_conversations", sa.Column("release_id", pg.UUID))
    op.alter_column("test_conversations", "created_by", nullable=True)
    op.alter_column("runtime_receipts", "created_by", nullable=True)
    for column, target in (("api_key_id", "agent_api_keys"), ("release_id", "releases")):
        op.create_foreign_key(
            f"fk_conversations_{column}",
            "test_conversations",
            target,
            ["organization_id", "agent_id", column],
            ["organization_id", "agent_id", "id"],
        )
    op.create_check_constraint(
        "ck_conversations_principal",
        "test_conversations",
        "(created_by IS NOT NULL AND api_key_id IS NULL AND release_id IS NULL) OR "
        "(created_by IS NULL AND api_key_id IS NOT NULL AND release_id IS NOT NULL "
        "AND http_mode = 'live')",
    )
    op.execute(
        "CREATE POLICY api_key_lookup ON agent_api_keys FOR SELECT TO dialogentis_app "
        "USING (token_hash = NULLIF(current_setting('app.presented_api_key_hash', true), ''))"
    )


def downgrade() -> None:
    op.execute("DROP POLICY api_key_lookup ON agent_api_keys")
    op.drop_table("runtime_rate_buckets")
    op.execute("ALTER TABLE test_conversations NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE runtime_receipts NO FORCE ROW LEVEL SECURITY")
    op.execute("DELETE FROM test_conversations WHERE api_key_id IS NOT NULL")
    op.execute("DELETE FROM runtime_receipts WHERE created_by IS NULL")
    op.execute("ALTER TABLE test_conversations FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE runtime_receipts FORCE ROW LEVEL SECURITY")
    op.drop_constraint("ck_conversations_principal", "test_conversations", type_="check")
    for column in ("api_key_id", "release_id"):
        op.drop_constraint(f"fk_conversations_{column}", "test_conversations", type_="foreignkey")
        op.drop_column("test_conversations", column)
    op.alter_column("test_conversations", "created_by", nullable=False)
    op.alter_column("runtime_receipts", "created_by", nullable=False)
