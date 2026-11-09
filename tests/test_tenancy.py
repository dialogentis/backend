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


def test_job_recovery_upgrade_preserves_existing_job():
    from sqlalchemy import create_engine

    config = Config("alembic.ini")
    url = Settings().database_url
    assert url is not None
    engine = create_engine(url.get_secret_value().replace("postgresql://", "postgresql+psycopg://"))
    tenant, job_id, event_id, training_id, training_event_id = (uuid4() for _ in range(5))
    try:
        command.downgrade(config, "0004")
        with engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.organization_id', :tenant, true)"),
                {"tenant": str(tenant)},
            )
            connection.execute(
                text(
                    "INSERT INTO jobs (id, organization_id, kind, status, attempts, "
                    "cancel_requested) "
                    "VALUES (:id, :tenant, 'existing', 'queued', 0, false)"
                ),
                {"id": job_id, "tenant": tenant},
            )
            connection.execute(
                text(
                    "INSERT INTO job_outbox (id, organization_id, job_id) "
                    "VALUES (:id, :tenant, :job)"
                ),
                {"id": event_id, "tenant": tenant, "job": job_id},
            )
            connection.execute(
                text(
                    "INSERT INTO jobs (id, organization_id, kind, status, attempts, "
                    "cancel_requested) "
                    "VALUES (:id, :tenant, 'agent.training', 'queued', 0, false)"
                ),
                {"id": training_id, "tenant": tenant},
            )
            connection.execute(
                text(
                    "INSERT INTO job_outbox (id, organization_id, job_id) "
                    "VALUES (:id, :tenant, :job)"
                ),
                {"id": training_event_id, "tenant": tenant, "job": training_id},
            )
        command.upgrade(config, "head")
        with engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.organization_id', :tenant, true)"),
                {"tenant": str(tenant)},
            )
            row = connection.execute(
                text(
                    "SELECT status, delivery_recoveries, execution_token, next_retry_at "
                    "FROM jobs WHERE id = :id"
                ),
                {"id": job_id},
            ).one()
            assert row == ("queued", 0, None, None)
            assert (
                connection.scalar(
                    text("SELECT job_id FROM job_outbox WHERE id = :id"), {"id": event_id}
                )
                == job_id
            )
            assert connection.execute(
                text(
                    "SELECT j.queue, o.queue FROM jobs j JOIN job_outbox o ON o.job_id = j.id "
                    "WHERE j.id = :id"
                ),
                {"id": training_id},
            ).one() == ("training", "training")
    finally:
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
