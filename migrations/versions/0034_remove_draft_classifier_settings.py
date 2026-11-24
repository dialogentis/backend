"""Remove retired classifier settings from editable drafts."""

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_drafts NO FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        UPDATE agent_drafts
        SET configuration = configuration - ARRAY['nlu_settings', 'quality_settings']
        WHERE configuration ?| ARRAY['nlu_settings', 'quality_settings']
        """
    )
    op.execute("ALTER TABLE agent_drafts FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # Previous application versions supplied defaults for both removed fields.
    pass
