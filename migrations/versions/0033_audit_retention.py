"""Support cursor pagination and discover organizations with expiring audit rows."""

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_audit_events_timeline", "audit_events", ["organization_id", "created_at", "id"]
    )
    op.execute("GRANT SELECT (organization_id) ON audit_events TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY audit_discovery ON audit_events FOR SELECT TO dialogentis_dispatcher "
        "USING (true)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY audit_discovery ON audit_events")
    op.execute("REVOKE SELECT (organization_id) ON audit_events FROM dialogentis_dispatcher")
    op.drop_index("ix_audit_events_timeline", table_name="audit_events")
