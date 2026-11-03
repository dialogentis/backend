from alembic import context
from sqlalchemy import create_engine

import dialogentis.auth.models  # noqa: F401 - register auth tables for metadata comparison
import dialogentis.jobs.models  # noqa: F401 - register job tables
import dialogentis.organizations  # noqa: F401 - register organization tables
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
