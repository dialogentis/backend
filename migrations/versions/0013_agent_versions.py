"""Immutable tenant-scoped training snapshots."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_versions",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("organization_id", pg.UUID, sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("agent_id", pg.UUID, nullable=False),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column("source_revision", sa.Integer, nullable=False),
        sa.Column("snapshot", pg.JSONB, nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("nlu_hash", sa.String(64), nullable=False),
        sa.Column("dataset_warnings", pg.JSONB, nullable=False),
        sa.Column("warnings_truncated", sa.Boolean, nullable=False),
        sa.Column("job_id", pg.UUID, nullable=False, unique=True),
        sa.Column("queue", sa.String(16), nullable=False, server_default="training"),
        sa.Column("idempotency_key_hash", sa.String(64), nullable=False),
        sa.Column("created_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("model_artifact_id", pg.UUID),
        sa.Column("validation_report", pg.JSONB),
        sa.UniqueConstraint("organization_id", "id"),
        sa.UniqueConstraint("organization_id", "agent_id", "id"),
        sa.UniqueConstraint("organization_id", "agent_id", "number"),
        sa.UniqueConstraint("organization_id", "agent_id", "created_by", "idempotency_key_hash"),
        sa.ForeignKeyConstraint(
            ["organization_id", "agent_id"], ["agents.organization_id", "agents.id"]
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "job_id", "queue"],
            ["jobs.organization_id", "jobs.id", "jobs.queue"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_artifact_id"], ["artifacts.organization_id", "artifacts.id"]
        ),
        sa.CheckConstraint("number > 0 AND source_revision > 0"),
        sa.CheckConstraint("queue = 'training'"),
        sa.CheckConstraint(
            "config_hash ~ '^[0-9a-f]{64}$' AND nlu_hash ~ '^[0-9a-f]{64}$' "
            "AND idempotency_key_hash ~ '^[0-9a-f]{64}$'"
        ),
        sa.CheckConstraint("(model_artifact_id IS NULL) = (validation_report IS NULL)"),
    )
    op.create_index("ix_agent_versions_organization_id", "agent_versions", ["organization_id"])
    op.create_index(
        "ix_agent_versions_nlu_hash", "agent_versions", ["organization_id", "agent_id", "nlu_hash"]
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE agent_versions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_versions FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON agent_versions USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )
    op.execute("""
        CREATE FUNCTION protect_agent_version() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'UPDATE' AND
                ROW(OLD.id, OLD.organization_id, OLD.agent_id, OLD.number, OLD.source_revision,
                    OLD.snapshot, OLD.config_hash, OLD.nlu_hash, OLD.dataset_warnings,
                    OLD.warnings_truncated, OLD.job_id, OLD.queue, OLD.idempotency_key_hash,
                    OLD.created_by, OLD.created_at)
                IS DISTINCT FROM
                ROW(NEW.id, NEW.organization_id, NEW.agent_id, NEW.number, NEW.source_revision,
                    NEW.snapshot, NEW.config_hash, NEW.nlu_hash, NEW.dataset_warnings,
                    NEW.warnings_truncated, NEW.job_id, NEW.queue, NEW.idempotency_key_hash,
                    NEW.created_by, NEW.created_at) THEN
                RAISE EXCEPTION 'Agent version snapshot is immutable' USING ERRCODE = '23514';
            END IF;
            IF TG_OP = 'UPDATE' AND OLD.model_artifact_id IS NOT NULL AND
                ROW(OLD.model_artifact_id, OLD.validation_report)
                IS DISTINCT FROM ROW(NEW.model_artifact_id, NEW.validation_report) THEN
                RAISE EXCEPTION 'Agent version result is immutable' USING ERRCODE = '23514';
            END IF;
            IF NEW.model_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts WHERE organization_id = NEW.organization_id
                AND id = NEW.model_artifact_id AND kind = 'model' AND status = 'ready'
            ) THEN
                RAISE EXCEPTION 'Model artifact is not ready' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$
    """)
    op.execute(
        "CREATE TRIGGER protect_agent_version BEFORE INSERT OR UPDATE ON agent_versions "
        "FOR EACH ROW EXECUTE FUNCTION protect_agent_version()"
    )


def downgrade() -> None:
    op.drop_table("agent_versions")
    op.execute("DROP FUNCTION protect_agent_version()")
