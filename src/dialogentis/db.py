from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def session_factory(url: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        url.replace("postgresql://", "postgresql+psycopg://"), pool_pre_ping=True
    )
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def unit_of_work(
    factory: async_sessionmaker[AsyncSession],
    *,
    organization_id: UUID | None = None,
    user_id: UUID | None = None,
) -> AsyncGenerator[AsyncSession]:
    """Transaction-local context also resets safely on rollback and pool reuse.

    Callers must check authentication and membership before selecting an organization.
    No method for disabling row-level security is provided.
    """
    async with factory() as session, session.begin():
        for key, value in [("app.organization_id", organization_id), ("app.user_id", user_id)]:
            await session.execute(
                text("SELECT set_config(:key, :value, true)"),
                {"key": key, "value": str(value) if value else ""},
            )
        yield session
