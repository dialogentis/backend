import asyncio
import os

import pytest
from fastapi import HTTPException
from sqlalchemy import delete

from dialogentis.db import unit_of_work
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import Membership
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.models import TestConversation as ConversationRecord
from dialogentis.runtime.test_chat import TestChatService as ChatService
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent, populate, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_real_model_chat_and_revoked_access_during_execution(app, monkeypatch):
    async with client_for(app) as owner, client_for(app) as viewer:
        user_id, org_id, agent_id = await create_agent(app, owner)
        viewer_id, _, _ = await create_agent(app, viewer)
        await populate(app, org_id, agent_id)
        version = (await prepare(app, user_id, org_id, agent_id, 1, "real-test-chat")).version
        version_id, job_id = version.id, version.job_id
        await process_job(app.state.session_factory, job_id, org_id, HANDLERS, "training")
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=viewer_id, role="viewer"))
        executor = RuntimeExecutor()
        service = ChatService(app.state.session_factory, app.state.settings, executor)
        started = await service.start(viewer_id, org_id, agent_id, version_id)
        assert started.revision == 1 and started.status == "waiting_input"
        result = await service.send(viewer_id, org_id, started.conversation_id, 1, "🛸🛸")
        assert result.revision == 2 and "🛸🛸" not in repr(result)
        with pytest.raises(HTTPException) as stale:
            await service.send(viewer_id, org_id, started.conversation_id, 1, "новый вопрос")
        assert stale.value.status_code == 409
        with pytest.raises(HTTPException) as wrong_user:
            await service.send(user_id, org_id, started.conversation_id, 2, "новый вопрос")
        assert wrong_user.value.status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, started.conversation_id)
            assert (
                row is not None
                and row.checkpoint is not None
                and "🛸🛸".encode() not in row.checkpoint
            )

        ready, release = asyncio.Event(), asyncio.Event()
        real_execute = executor.execute

        async def paused(**kwargs):
            ready.set()
            await release.wait()
            return await real_execute(**kwargs)

        monkeypatch.setattr(executor, "execute", paused)
        task = asyncio.create_task(
            service.send(viewer_id, org_id, started.conversation_id, 2, "🧩🧩")
        )
        await asyncio.wait_for(ready.wait(), 10)
        try:
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                await db.execute(
                    delete(Membership).where(
                        Membership.organization_id == org_id, Membership.user_id == viewer_id
                    )
                )
        finally:
            release.set()
        with pytest.raises(HTTPException) as revoked:
            await task
        assert revoked.value.status_code == 404
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            row = await db.get(ConversationRecord, started.conversation_id)
            assert row is not None and row.revision == 2 and row.processing_token is not None
