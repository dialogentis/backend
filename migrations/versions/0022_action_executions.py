"""Persist fenced, tenant-scoped external action executions."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_receipt_tenant_id_conversation",
        "runtime_receipts",
        ["organization_id", "id", "conversation_id"],
    )
    op.create_table(
        "action_executions",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("conversation_id", pg.UUID, nullable=False),
        sa.Column("receipt_id", pg.UUID, nullable=False),
        sa.Column("integration_id", pg.UUID, nullable=False),
        sa.Column("node_id", pg.UUID, nullable=False),
        sa.Column("processing_token", pg.UUID, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("method", sa.String(6), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("http_status", sa.Integer),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["organization_id", "receipt_id", "conversation_id"],
            [
                "runtime_receipts.organization_id",
                "runtime_receipts.id",
                "runtime_receipts.conversation_id",
            ],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "integration_id"],
            ["integrations.organization_id", "integrations.id"],
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.UniqueConstraint(
            "organization_id", "receipt_id", "ordinal", name="uq_action_turn_ordinal"
        ),
        sa.CheckConstraint("ordinal BETWEEN 1 AND 3"),
        sa.CheckConstraint("method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')"),
        sa.CheckConstraint(
            "status IN ('started', 'succeeded', 'http_error', 'timeout', 'invalid_response', "
            "'unknown_result', 'network_error', 'address_blocked', 'request_not_allowed', "
            "'dns_error', 'invalid_request', 'invalid_credential')"
        ),
        sa.CheckConstraint(
            "(status = 'started' AND finished_at IS NULL AND http_status IS NULL) OR "
            "(status <> 'started' AND finished_at IS NOT NULL)"
        ),
        sa.CheckConstraint("http_status IS NULL OR http_status BETWEEN 100 AND 599"),
        sa.CheckConstraint(
            "(status = 'succeeded' AND http_status IS NOT NULL "
            "AND http_status BETWEEN 200 AND 299) "
            "OR (status <> 'succeeded' AND http_status IS NULL)"
        ),
    )
    op.create_index(
        "ix_action_executions_integration",
        "action_executions",
        ["organization_id", "integration_id"],
    )
    op.create_index(
        "ix_action_executions_conversation_status",
        "action_executions",
        ["organization_id", "conversation_id", "status"],
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE action_executions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE action_executions FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON action_executions USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )


def downgrade() -> None:
    op.drop_table("action_executions")
    op.drop_constraint("uq_receipt_tenant_id_conversation", "runtime_receipts", type_="unique")
