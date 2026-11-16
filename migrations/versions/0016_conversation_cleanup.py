"""Allow maintenance to discover tenants with test conversations."""

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("GRANT SELECT (organization_id) ON test_conversations TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY dispatcher_conversation_tenants ON test_conversations FOR SELECT "
        "TO dialogentis_dispatcher USING (true)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY dispatcher_conversation_tenants ON test_conversations")
    op.execute("REVOKE SELECT (organization_id) ON test_conversations FROM dialogentis_dispatcher")
