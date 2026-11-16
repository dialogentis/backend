"""Durable tenant-scoped message receipts; text payloads are encrypted."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_receipts",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column(
            "organization_id",
            pg.UUID,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("conversation_id", pg.UUID, nullable=False),
        sa.Column("created_by", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("credential_hash", sa.String(64), nullable=False),
        sa.Column("client_message_id", pg.UUID, nullable=False),
        sa.Column("idempotency_key_hash", sa.String(64), nullable=False),
        sa.Column("request_digest", sa.String(64), nullable=False),
        sa.Column("key_version", sa.BigInteger, nullable=False),
        sa.Column("processing_token", pg.UUID, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("revision", sa.Integer),
        sa.Column("runtime_status", sa.String(16)),
        sa.Column("encrypted_response", sa.LargeBinary),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "organization_id", "conversation_id", "client_message_id", name="uq_receipt_message"
        ),
        sa.UniqueConstraint(
            "organization_id", "conversation_id", "idempotency_key_hash", name="uq_receipt_key"
        ),
        sa.CheckConstraint(
            "credential_hash ~ '^[0-9a-f]{64}$' AND idempotency_key_hash ~ '^[0-9a-f]{64}$' "
            "AND request_digest ~ '^[0-9a-f]{64}$'"
        ),
        sa.CheckConstraint("key_version > 0 AND key_version <= 4294967295"),
        sa.CheckConstraint("status IN ('processing', 'completed', 'unknown')"),
        sa.CheckConstraint(
            "(status = 'completed' AND revision IS NOT NULL AND revision > 0 "
            "AND runtime_status IS NOT NULL "
            "AND runtime_status IN ('waiting_input', 'completed', 'failed')) "
            "OR (status <> 'completed' AND revision IS NULL AND runtime_status IS NULL "
            "AND encrypted_response IS NULL)"
        ),
        sa.CheckConstraint(
            "encrypted_response IS NULL OR "
            "(octet_length(encrypted_response) BETWEEN 36 AND 2097188 "
            "AND substring(encrypted_response from 1 for 4) = decode('44475231', 'hex'))"
        ),
    )
    op.create_index(
        "ix_runtime_receipts_org_expiry", "runtime_receipts", ["organization_id", "expires_at"]
    )
    tenant = "NULLIF(current_setting('app.organization_id', true), '')::uuid"
    op.execute("ALTER TABLE runtime_receipts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE runtime_receipts FORCE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON runtime_receipts USING (organization_id = {tenant}) "
        f"WITH CHECK (organization_id = {tenant})"
    )
    op.execute("GRANT SELECT (organization_id) ON runtime_receipts TO dialogentis_dispatcher")
    op.execute(
        "CREATE POLICY dispatcher_receipt_tenants ON runtime_receipts FOR SELECT "
        "TO dialogentis_dispatcher USING (true)"
    )


def downgrade() -> None:
    op.drop_table("runtime_receipts")
