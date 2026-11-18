"""Store retained conversation history separately from live state and replay."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_history",
        sa.Column("organization_id", pg.UUID, primary_key=True),
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("version_id", pg.UUID, nullable=False),
        sa.Column("release_id", pg.UUID),
        sa.Column("channel_type", sa.String(8), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"],
            ["agents.organization_id", "agents.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("channel_type IN ('test','api','widget')"),
        sa.CheckConstraint("status IN ('waiting_input','completed','failed','expired')"),
    )
    op.create_index(
        "ix_history_created", "conversation_history", ["organization_id", "created_at", "id"]
    )
    op.create_index(
        "ix_history_activity", "conversation_history", ["organization_id", "last_activity_at"]
    )
    op.create_table(
        "conversation_history_turns",
        sa.Column("organization_id", pg.UUID, primary_key=True),
        sa.Column("conversation_id", pg.UUID, primary_key=True),
        sa.Column("sequence", sa.Integer, primary_key=True),
        sa.Column("turn_id", pg.UUID, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("intent_id", pg.UUID),
        sa.Column("error_code", sa.String(64)),
        sa.Column("trace", pg.JSONB, nullable=False),
        sa.Column("encrypted_messages", sa.LargeBinary),
        sa.ForeignKeyConstraint(
            ["organization_id", "conversation_id"],
            ["conversation_history.organization_id", "conversation_history.id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("sequence >= 0"),
    )
    op.create_index(
        "ix_history_turn_created", "conversation_history_turns", ["organization_id", "created_at"]
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    for table in ("conversation_history", "conversation_history_turns"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING (organization_id = {tenant}) "
            f"WITH CHECK (organization_id = {tenant})"
        )
    op.execute("GRANT SELECT (organization_id) ON conversation_history TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY dispatcher_history_tenants ON conversation_history FOR SELECT "
        "TO dialogentis_dispatcher USING (true)"
    )


def downgrade() -> None:
    op.drop_table("conversation_history_turns")
    op.drop_table("conversation_history")
