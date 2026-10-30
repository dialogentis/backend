import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from dialogentis.config import Settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.models import Agent, AgentDraft, Organization

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Run with Compose test profile",
    ),
]


def test_migrations_can_run_twice_without_schema_drift():
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from dialogentis.models import Base

    config = Config("alembic.ini")
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    url = Settings().migration_database_url
    assert url is not None
    engine = create_engine(url.get_secret_value().replace("postgresql://", "postgresql+psycopg://"))
    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
    engine.dispose()


async def test_rls_composite_foreign_keys_and_pool_reset():
    url = Settings().database_url
    assert url is not None
    factory = session_factory(url.get_secret_value())
    first_org, second_org = uuid4(), uuid4()
    first_agent, second_agent = uuid4(), uuid4()
    for organization_id, agent_id in [(first_org, first_agent), (second_org, second_agent)]:
        async with unit_of_work(factory, organization_id=organization_id) as session:
            role = (
                await session.execute(
                    text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
                )
            ).one()
            assert role == (False, False)
            session.add(Organization(id=organization_id, name="Isolation test", slug=str(uuid4())))
            await session.flush()
            session.add(
                Agent(id=agent_id, organization_id=organization_id, name="Agent", slug="same-slug")
            )
    async with unit_of_work(factory) as session:
        assert (await session.scalars(select(Agent))).all() == []
        assert (await session.scalars(select(Organization))).all() == []
    async with unit_of_work(factory, organization_id=first_org) as session:
        assert (await session.scalars(select(Agent.id))).all() == [first_agent]
        assert await session.get(Agent, second_agent) is None
        owner = (
            await session.execute(
                text("SELECT tableowner FROM pg_tables WHERE tablename = 'agents'")
            )
        ).scalar_one()
        assert owner == "dialogentis_migrator"
    with pytest.raises(IntegrityError):
        async with unit_of_work(factory, organization_id=first_org) as session:
            session.add(
                AgentDraft(organization_id=first_org, agent_id=second_agent, configuration={})
            )
    with pytest.raises(DBAPIError):
        async with unit_of_work(factory, organization_id=first_org) as session:
            session.add(Agent(organization_id=second_org, name="Forbidden", slug="cross-tenant"))
    async with unit_of_work(factory) as session:
        assert (await session.scalars(select(Agent))).all() == []
    async with unit_of_work(factory, organization_id=second_org) as session:
        assert (await session.scalars(select(Agent.id))).all() == [second_agent]
    # Tests do not need superuser cleanup; each tenant may remove its own temporary records.
    for organization_id in [first_org, second_org]:
        async with unit_of_work(factory, organization_id=organization_id) as session:
            await session.execute(
                text("DELETE FROM agents WHERE organization_id = :id"), {"id": organization_id}
            )
            await session.execute(
                text("DELETE FROM organizations WHERE id = :id"), {"id": organization_id}
            )
    await factory.kw["bind"].dispose()
