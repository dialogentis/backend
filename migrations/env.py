from alembic import context
from sqlalchemy import create_engine

import dialogentis.artifact_records  # noqa: F401 - register artifact tables
import dialogentis.auth.models  # noqa: F401 - register auth tables for metadata comparison
import dialogentis.channels.api_key_models  # noqa: F401 - register API keys
import dialogentis.channels.widget_models  # noqa: F401 - register website widgets
import dialogentis.datasets.export_models  # noqa: F401 - register export tables
import dialogentis.datasets.import_models  # noqa: F401 - register import tables
import dialogentis.integrations.ledger_models  # noqa: F401 - register action executions
import dialogentis.integrations.models  # noqa: F401 - register integration tables
import dialogentis.jobs.models  # noqa: F401 - register job tables
import dialogentis.organizations  # noqa: F401 - register organization tables
import dialogentis.runtime.history_export_models  # noqa: F401 - register history exports
import dialogentis.runtime.history_models  # noqa: F401 - register conversation history
import dialogentis.runtime.models  # noqa: F401 - register conversation tables
import dialogentis.runtime.rate_limits  # noqa: F401 - register rate buckets
import dialogentis.runtime.receipt_models  # noqa: F401 - register receipt tables
import dialogentis.versions.models  # noqa: F401 - register immutable version tables
import dialogentis.versions.release_models  # noqa: F401 - register release tables
from dialogentis.config import get_settings
from dialogentis.models import Base

config = get_settings()
if config.migration_database_url is None:
    raise ValueError("MIGRATION_DATABASE_URL is required")
engine = create_engine(
    config.migration_database_url.get_secret_value().replace(
        "postgresql://",
        "postgresql+psycopg://",
    )
)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
