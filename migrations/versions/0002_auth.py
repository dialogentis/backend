"""Global sessions, one-time tokens and durable encrypted mail delivery."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    for name in ["sessions", "auth_tokens"]:
        columns = [
            sa.Column("id", pg.UUID, primary_key=True),
            sa.Column("user_id", pg.UUID, sa.ForeignKey("users.id"), nullable=False),
            sa.Column("token_hash", sa.String(64), unique=True, nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        ]
        if name == "sessions":
            columns.append(
                sa.Column("absolute_expires_at", sa.DateTime(timezone=True), nullable=False)
            )
        else:
            columns.extend(
                [
                    sa.Column("purpose", sa.String(16), nullable=False),
                    sa.Column("consumed_at", sa.DateTime(timezone=True)),
                ]
            )
        op.create_table(name, *columns)
        op.create_index(f"ix_{name}_user_id", name, ["user_id"])
    op.create_table(
        "auth_rate_events",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("subject_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_auth_rate_subject_time", "auth_rate_events", ["subject_hash", "created_at"])
    op.create_table(
        "email_outbox",
        sa.Column("id", pg.UUID, primary_key=True),
        sa.Column("encrypted_payload", sa.Text, nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
    )


def downgrade():
    for name in ["email_outbox", "auth_rate_events", "auth_tokens", "sessions"]:
        op.drop_table(name)
