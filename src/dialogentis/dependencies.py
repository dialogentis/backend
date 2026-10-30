from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from dialogentis.db import unit_of_work


async def get_db(request: Request) -> AsyncGenerator[AsyncSession]:
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(503, "База данных недоступна")
    async with unit_of_work(factory) as session:
        yield session


Database = Annotated[AsyncSession, Depends(get_db, scope="function")]
