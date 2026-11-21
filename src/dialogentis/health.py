import asyncio
from functools import cache
from pathlib import Path
from typing import Literal
from uuid import UUID

import psycopg
from alembic.script import ScriptDirectory
from pydantic import BaseModel

from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings

type Check = Literal["ok", "unavailable"]


class RuntimeChecks(BaseModel):
    database: Check = "unavailable"
    schema_revision: Check = "unavailable"
    artifacts: Check = "unavailable"


class RuntimeReadiness(BaseModel):
    status: Literal["ready", "unavailable"]
    checks: RuntimeChecks
    request_id: UUID


@cache
def expected_schema() -> str | None:
    directory = Path(__file__).resolve().parents[2] / "migrations"
    return ScriptDirectory(str(directory)).get_current_head() if directory.is_dir() else None


async def runtime_checks(config: Settings) -> RuntimeChecks:
    checks = RuntimeChecks()
    if config.database_url:
        try:
            async with asyncio.timeout(3):
                url = config.database_url.get_secret_value().replace(
                    "postgresql+psycopg", "postgresql"
                )
                async with await psycopg.AsyncConnection.connect(url) as connection:
                    await connection.execute("SELECT 1")
                    checks.database = "ok"
                    heads = await (
                        await connection.execute("SELECT version_num FROM alembic_version")
                    ).fetchall()
                    if (expected := expected_schema()) is not None and heads == [(expected,)]:
                        checks.schema_revision = "ok"
        except TimeoutError, psycopg.Error:
            pass
    if await asyncio.to_thread(ArtifactStorage(config.artifact_root).available):
        checks.artifacts = "ok"
    return checks
