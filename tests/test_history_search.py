import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete

from dialogentis.db import unit_of_work
from dialogentis.runtime.history import PURPOSE
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.rate_limits import RateBucket
from dialogentis.runtime.receipts import ReceiptIdentity, ReceiptPayload, seal_receipt
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_versions import create_agent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_search_continues_inside_a_dialogue_and_respects_privacy(app, monkeypatch):
    monkeypatch.setattr("dialogentis.runtime.history_search.SCAN_TURNS", 2)
    async with client_for(app) as owner, client_for(app) as other:
        _, org_id, agent_id = await create_agent(app, owner)
        _, other_org, _ = await create_agent(app, other)
        now = datetime.now(UTC)
        history = f"/api/v1/organizations/{org_id}/conversations"
        search = f"{history}/search"
        org = f"/api/v1/organizations/{org_id}"
        newest, middle, oldest = uuid4(), uuid4(), uuid4()
        selected_intent = uuid4()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            for offset, item in enumerate((newest, middle, oldest)):
                created = now - timedelta(minutes=1 + offset)
                db.add(
                    ConversationHistory(
                        id=item,
                        organization_id=org_id,
                        agent_id=agent_id,
                        version_id=uuid4(),
                        channel_type="widget",
                        status="completed",
                        created_at=created,
                        last_activity_at=created,
                    )
                )
            await db.flush()
            for item, texts in (
                (newest, ("Начало", "Уточнение", "Окончание")),
                (middle, ("Номер заказа [скрыто]", "Номер заказа", "[скрыто]")),
                (oldest, ("Номер заказа",)),
            ):
                for seq, text in enumerate(texts):
                    turn = uuid4()
                    encrypted = seal_receipt(
                        ReceiptPayload(
                            revision=seq + 1,
                            status="completed",
                            messages=(),
                            trace=(),
                            input_text=text,
                        ),
                        ReceiptIdentity(org_id, item, turn),
                        app.state.settings,
                        purpose=PURPOSE,
                    )
                    db.add(
                        HistoryTurn(
                            organization_id=org_id,
                            conversation_id=item,
                            sequence=seq,
                            turn_id=turn,
                            created_at=now - timedelta(seconds=30),
                            intent_id=selected_intent if item == middle else None,
                            trace=[],
                            encrypted_messages=encrypted,
                        )
                    )
            turn = uuid4()
            encrypted = seal_receipt(
                ReceiptPayload(
                    revision=4,
                    status="completed",
                    messages=("Просроченный текст",),
                    trace=(),
                ),
                ReceiptIdentity(org_id, newest, turn),
                app.state.settings,
                purpose=PURPOSE,
            )
            db.add(
                HistoryTurn(
                    organization_id=org_id,
                    conversation_id=newest,
                    sequence=3,
                    turn_id=turn,
                    created_at=now - timedelta(days=31),
                    trace=[],
                    encrypted_messages=encrypted,
                )
            )

        body = {"text": "ЗАКАЗА", "filters": {"channel_type": "widget"}, "limit": 1}
        first = await owner.post(search, json=body)
        assert first.status_code == 200, first.text
        assert first.json()["items"] == []
        assert first.json()["next_cursor"]
        second = await owner.post(search, json={**body, "cursor": first.json()["next_cursor"]})
        assert [item["id"] for item in second.json()["items"]] == [str(middle)]
        third = await owner.post(search, json={**body, "cursor": second.json()["next_cursor"]})
        assert [item["id"] for item in third.json()["items"]] == [str(oldest)]
        assert third.json()["next_cursor"] is None

        query = {**body, "filters": {"intent_id": str(selected_intent)}}
        filtered = await other.post(search, json=query)
        assert filtered.status_code == 404
        own = await other.post(
            f"/api/v1/organizations/{other_org}/conversations/search", json=query
        )
        assert own.json() == {"items": [], "next_cursor": None}
        assert [item["id"] for item in (await owner.post(search, json=query)).json()["items"]] == [
            str(middle)
        ]
        assert (await owner.post(search, json={**body, "cursor": "malformed"})).status_code == 422

        async def set_text(enabled, purge=False):
            etag = (await owner.get(org)).headers["etag"]
            response = await owner.patch(
                org,
                json={"store_message_text": enabled, "purge_existing_text": purge},
                headers={"If-Match": etag},
            )
            assert response.status_code == 200

        await set_text(False)
        assert (await owner.post(search, json=body)).status_code == 409
        await set_text(True)
        for invisible in ("скрыто", "Просроченный текст"):
            await clear_search_rate(app, org_id)
            cursor = None
            for _ in range(4):
                response = await owner.post(search, json={"text": invisible, "cursor": cursor})
                assert response.status_code == 200, response.text
                assert response.json()["items"] == []
                cursor = response.json()["next_cursor"]
                if cursor is None:
                    break
            assert cursor is None
            await clear_search_rate(app, org_id)
        await set_text(False, purge=True)
        await set_text(True)
        assert (await owner.post(search, json=body)).json() == {"items": [], "next_cursor": None}


async def clear_search_rate(app, org_id):
    async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
        await db.execute(delete(RateBucket).where(RateBucket.organization_id == org_id))
