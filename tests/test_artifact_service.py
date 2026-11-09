import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError, IntegrityError

from dialogentis.artifact_records import Artifact
from dialogentis.artifact_service import (
    ArtifactQuotaExceeded,
    ArtifactUnavailable,
    cleanup_expired,
    cleanup_orphans,
    read_artifact,
    reserve_artifact,
    write_artifact,
)
from dialogentis.artifacts import ArtifactStorage
from dialogentis.config import Settings, get_settings
from dialogentis.db import session_factory, unit_of_work
from dialogentis.jobs.dispatcher import dispatcher_factory
from dialogentis.jobs.maintenance import reconcile_once
from dialogentis.models import Organization

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


@pytest.fixture
async def context(tmp_path):
    settings = Settings(artifact_quota_bytes=10, temporary_artifact_quota_bytes=10)
    assert settings.database_url is not None
    factory = session_factory(settings.database_url.get_secret_value())
    first, second = uuid4(), uuid4()
    for org_id in (first, second):
        async with unit_of_work(factory, organization_id=org_id) as db:
            db.add(Organization(id=org_id, name="Test", slug=f"org-{uuid4()}"))
    try:
        yield factory, ArtifactStorage(tmp_path), settings, first, second
    finally:
        for org_id in (first, second):
            async with unit_of_work(factory, organization_id=org_id) as db:
                await db.execute(delete(Artifact))
                await db.execute(delete(Organization).where(Organization.id == org_id))
        await factory.kw["bind"].dispose()


async def test_concurrent_reservations_obey_quota_and_row_security(context):
    factory, storage, settings, first, second = context

    async def reserve():
        async with unit_of_work(factory, organization_id=first) as db:
            return (
                await reserve_artifact(db, first, kind="temporary", max_bytes=6, settings=settings)
            ).id

    results = await asyncio.gather(reserve(), reserve(), return_exceptions=True)
    assert sum(isinstance(result, ArtifactQuotaExceeded) for result in results) == 1
    artifact_id = next(result for result in results if not isinstance(result, BaseException))
    async with unit_of_work(factory) as db:
        assert (await db.scalars(select(Artifact))).all() == []
    async with unit_of_work(factory, organization_id=second) as db:
        assert await db.get(Artifact, artifact_id) is None
        with pytest.raises(ArtifactUnavailable):
            await read_artifact(db, storage, second, artifact_id, max_bytes=10)
    async with unit_of_work(factory, organization_id=first) as db:
        with pytest.raises(ArtifactUnavailable):
            await read_artifact(db, storage, first, artifact_id, max_bytes=10)


async def test_finalize_expiry_and_separate_model_quota(context):
    factory, storage, settings, first, _ = context
    async with unit_of_work(factory, organization_id=first) as db:
        item = await reserve_artifact(db, first, kind="temporary", max_bytes=8, settings=settings)
        file_id = item.id
        model = await reserve_artifact(db, first, kind="model", max_bytes=10, settings=settings)
        model_id = model.id
    async with unit_of_work(factory, organization_id=first) as db:
        item = await write_artifact(db, storage, first, file_id, [b"data"])
        assert item.size_bytes == 4 and item.status == "ready"
    async with unit_of_work(factory, organization_id=first) as db:
        assert await read_artifact(db, storage, first, file_id, max_bytes=10) == b"data"
        await write_artifact(db, storage, first, model_id, [b"model"])
    async with unit_of_work(factory, organization_id=first) as db:
        extra = await reserve_artifact(db, first, kind="temporary", max_bytes=6, settings=settings)
        extra_id = extra.id
    async with unit_of_work(factory, organization_id=first) as db:
        item = await db.get(Artifact, file_id)
        extra = await db.get(Artifact, extra_id)
        assert item is not None and extra is not None
        item.expires_at = extra.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    async with unit_of_work(factory, organization_id=first) as db:
        with pytest.raises(ArtifactUnavailable):
            await read_artifact(db, storage, first, file_id, max_bytes=10)
        assert await cleanup_expired(db, storage, first) == 2
        assert await db.get(Artifact, file_id) is None
        assert await read_artifact(db, storage, first, model_id, max_bytes=10) == b"model"
    assert not (storage.root / str(first) / str(file_id)).exists()


async def test_rollback_file_is_safely_replaced_by_same_reservation(context):
    factory, storage, settings, first, _ = context
    async with unit_of_work(factory, organization_id=first) as db:
        artifact_id = (
            await reserve_artifact(db, first, kind="temporary", max_bytes=10, settings=settings)
        ).id
    with pytest.raises(RuntimeError):
        async with unit_of_work(factory, organization_id=first) as db:
            await write_artifact(db, storage, first, artifact_id, [b"old"])
            raise RuntimeError("rollback")
    async with unit_of_work(factory, organization_id=first) as db:
        item = await db.get(Artifact, artifact_id)
        assert item is not None and item.status == "reserved"
        await write_artifact(db, storage, first, artifact_id, [b"new"])
    async with unit_of_work(factory, organization_id=first) as db:
        assert await read_artifact(db, storage, first, artifact_id, max_bytes=10) == b"new"


async def test_cancellation_keeps_tenant_locked_until_writer_stops(context, monkeypatch):
    factory, storage, settings, first, _ = context
    async with unit_of_work(factory, organization_id=first) as db:
        artifact_id = (
            await reserve_artifact(db, first, kind="temporary", max_bytes=10, settings=settings)
        ).id
    started, finish = Event(), Event()
    original = storage.write

    def slow_write(*args, **kwargs):
        started.set()
        assert finish.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "write", slow_write)

    async def writing():
        async with unit_of_work(factory, organization_id=first) as db:
            await write_artifact(db, storage, first, artifact_id, [b"data"])

    async def close_org():
        async with unit_of_work(factory, organization_id=first) as db:
            org = await db.scalar(
                select(Organization).where(Organization.id == first).with_for_update()
            )
            assert org is not None
            org.deletion_requested_at = datetime.now(UTC)

    task = asyncio.create_task(writing())
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    closing = asyncio.create_task(close_org())
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), timeout=0.1)
        assert not task.done()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closing, timeout=3)


async def test_database_rejects_ready_artifact_without_checksum(context):
    factory, _, _, first, _ = context
    with pytest.raises(IntegrityError):
        async with unit_of_work(factory, organization_id=first) as db:
            db.add(
                Artifact(
                    id=uuid4(), organization_id=first, kind="model", status="ready", size_bytes=1
                )
            )
            await db.flush()


async def test_scheduled_cleanup_finds_expired_tenants_with_limited_dispatcher(
    context, monkeypatch
):
    factory, storage, settings, first, second = context
    monkeypatch.setenv("ARTIFACT_ROOT", str(storage.root))
    get_settings.cache_clear()
    async with unit_of_work(factory, organization_id=first) as db:
        artifact_id = (
            await reserve_artifact(db, first, kind="temporary", max_bytes=10, settings=settings)
        ).id
    async with unit_of_work(factory, organization_id=first) as db:
        artifact = await write_artifact(db, storage, first, artifact_id, [b"data"])
        artifact.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    orphan = uuid4()
    storage.write(second, orphan, [b"orphan"], max_bytes=10)
    orphan_path = storage.root / str(second) / str(orphan)
    old = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(orphan_path, (old, old))
    missing_org, missing_file = uuid4(), uuid4()
    storage.write(missing_org, missing_file, [b"unknown"], max_bytes=10)
    unknown_path = storage.root / str(missing_org) / str(missing_file)
    os.utime(unknown_path, (old, old))
    dispatch = dispatcher_factory(Path("/run/dialogentis-dispatcher/password"))
    try:
        async with dispatch() as db:
            assert first in (await db.scalars(select(Artifact.organization_id).distinct())).all()
        with pytest.raises(DBAPIError):
            async with dispatch() as db:
                await db.execute(select(Artifact.id))
        window = datetime(2025, 1, 1, tzinfo=UTC)
        assert await reconcile_once(factory, dispatch, window, frozenset())
        assert not await reconcile_once(factory, dispatch, window, frozenset())
        async with unit_of_work(factory, organization_id=first) as db:
            assert await db.get(Artifact, artifact_id) is None
        assert not (storage.root / str(first) / str(artifact_id)).exists()
        assert not orphan_path.exists()
        assert unknown_path.read_bytes() == b"unknown"
    finally:
        await dispatch.kw["bind"].dispose()
        get_settings.cache_clear()


async def test_orphan_cleanup_waits_for_inflight_write(context, monkeypatch):
    factory, storage, settings, first, _ = context
    async with unit_of_work(factory, organization_id=first) as db:
        artifact_id = (
            await reserve_artifact(db, first, kind="temporary", max_bytes=10, settings=settings)
        ).id
    orphan = uuid4()
    storage.write(first, orphan, [b"old"], max_bytes=10)
    orphan_path = storage.root / str(first) / str(orphan)
    old = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(orphan_path, (old, old))
    started, finish = Event(), Event()
    original = storage.write

    def slow_write(*args, **kwargs):
        started.set()
        assert finish.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "write", slow_write)

    async def writing():
        async with unit_of_work(factory, organization_id=first) as db:
            await write_artifact(db, storage, first, artifact_id, [b"current"])

    async def cleaning():
        async with unit_of_work(factory, organization_id=first) as db:
            return await cleanup_orphans(db, storage, first)

    writer = asyncio.create_task(writing())
    assert await asyncio.to_thread(started.wait, 2)
    cleaner = asyncio.create_task(cleaning())
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(cleaner), timeout=0.1)
    finally:
        finish.set()
    await asyncio.wait_for(writer, timeout=3)
    assert await asyncio.wait_for(cleaner, timeout=3) == 1
    async with unit_of_work(factory, organization_id=first) as db:
        assert await read_artifact(db, storage, first, artifact_id, max_bytes=10) == b"current"
    assert not orphan_path.exists()
