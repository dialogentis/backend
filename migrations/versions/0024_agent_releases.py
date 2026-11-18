"""Immutable releases and tenant-scoped version pins."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "releases",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("root_version_id", pg.UUID, nullable=False),
        sa.Column("dependency_manifest", pg.JSONB, nullable=False),
        sa.Column("created_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("reason", sa.String(500)),
        sa.UniqueConstraint("organization_id", "agent_id", "id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id", "root_version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
        ),
    )
    op.create_index(
        "ix_releases_agent_created", "releases", ["organization_id", "agent_id", "created_at", "id"]
    )
    op.create_table(
        "release_versions",
        sa.Column("organization_id", pg.UUID, primary_key=True),
        sa.Column("release_agent_id", pg.UUID, nullable=False),
        sa.Column("release_id", pg.UUID, primary_key=True),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("version_id", pg.UUID, primary_key=True),
        sa.ForeignKeyConstraint(
            ["organization_id", "release_agent_id", "release_id"],
            ["releases.organization_id", "releases.agent_id", "releases.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id", "version_id"],
            ["agent_versions.organization_id", "agent_versions.agent_id", "agent_versions.id"],
        ),
    )
    op.create_index(
        "ix_release_versions_agent", "release_versions", ["organization_id", "agent_id"]
    )
    op.add_column("agents", sa.Column("current_release_id", pg.UUID))
    op.create_foreign_key(
        "fk_agents_current_release",
        "agents",
        "releases",
        ["organization_id", "id", "current_release_id"],
        ["organization_id", "agent_id", "id"],
    )
    op.add_column(
        "audit_events", sa.Column("details", pg.JSONB, nullable=False, server_default="{}")
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    for table in ("releases", "release_versions"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} USING (organization_id = {tenant}) "
            f"WITH CHECK (organization_id = {tenant})"
        )
    op.execute("REVOKE UPDATE ON releases, release_versions FROM dialogentis_app")


def downgrade() -> None:
    op.drop_column("audit_events", "details")
    op.drop_constraint("fk_agents_current_release", "agents", type_="foreignkey")
    op.drop_column("agents", "current_release_id")
    op.drop_table("release_versions")
    op.drop_table("releases")
