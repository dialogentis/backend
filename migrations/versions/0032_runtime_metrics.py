"""Record deduplicated analytics events and tenant-scoped UTC buckets."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "metric_events",
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("event_key", sa.String(64), primary_key=True),
        sa.Column("bucket_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_metric_events_expiry", "metric_events", ["organization_id", "bucket_at"])
    op.create_table(
        "metric_hours",
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("bucket_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("root_agent_id", pg.UUID, primary_key=True),
        sa.Column("active_agent_id", pg.UUID, primary_key=True),
        sa.Column("channel", sa.String(8), primary_key=True),
        sa.Column("metric", sa.String(24), primary_key=True),
        sa.Column("item", sa.String(36), primary_key=True),
        sa.Column("count", sa.BigInteger, nullable=False),
        sa.Column("minute_counts", pg.JSONB, nullable=False),
        sa.CheckConstraint("channel IN ('test', 'api', 'widget')"),
        sa.CheckConstraint("count > 0"),
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    for table in ("metric_events", "metric_hours"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING (organization_id = {tenant}) "
            f"WITH CHECK (organization_id = {tenant})"
        )
    op.execute("GRANT SELECT (organization_id) ON metric_hours TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY metrics_discovery ON metric_hours FOR SELECT TO dialogentis_dispatcher "
        "USING (true)"
    )


def downgrade() -> None:
    op.drop_table("metric_hours")
    op.drop_table("metric_events")
