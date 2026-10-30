"""Invitation and ownership boundaries."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "uq_organization_owner",
        "memberships",
        ["organization_id"],
        unique=True,
        postgresql_where=sa.text("role = 'owner'"),
    )
    op.create_table(
        "invitations",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("token_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True)),
        sa.Column("revoked", sa.Boolean, nullable=False),
    )
    op.create_index("ix_invitations_organization_id", "invitations", ["organization_id"])
    op.create_table(
        "audit_events",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("actor_id", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_id", pg.UUID, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_organization_id", "audit_events", ["organization_id"])
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    user = "NULLIF(current_setting('app.user_id', true), '')::uuid"
    for table in ["invitations", "audit_events"]:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        read = f"organization_id = {tenant}"
        if table == "invitations":
            read += f" OR email = (SELECT email FROM users WHERE id = {user})"
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING ({read}) "
            f"WITH CHECK (organization_id = {tenant})"
        )
    op.execute(
        "CREATE POLICY membership_organization_read ON organizations FOR SELECT USING "
        "(EXISTS (SELECT 1 FROM memberships WHERE organization_id = organizations.id "
        f"AND user_id = {user}))"
    )


def downgrade():
    op.execute("DROP POLICY membership_organization_read ON organizations")
    op.drop_table("audit_events")
    op.drop_table("invitations")
    op.drop_index("uq_organization_owner", table_name="memberships")
