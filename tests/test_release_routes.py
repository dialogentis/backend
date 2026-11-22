import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import DBAPIError

from dialogentis.analytics_models import MetricHour
from dialogentis.db import unit_of_work
from dialogentis.deletion_worker import purge_tenant
from dialogentis.jobs.tasks import process_job
from dialogentis.jobs.training_tasks import HANDLERS
from dialogentis.models import AgentDraft, Membership, Organization
from dialogentis.organizations import AuditEvent
from dialogentis.runtime.history_models import ConversationHistory, HistoryTurn
from dialogentis.runtime.rate_limits import RateBucket
from dialogentis.runtime.receipt_models import RuntimeReceipt
from dialogentis.training import worker as training_worker
from dialogentis.versions.release_models import Release, ReleaseVersion
from tests.test_auth import client_for
from tests.test_exports import app as app
from tests.test_nlu_intents import training_dataset
from tests.test_runtime_handoff import call_flows
from tests.test_versions import create_agent, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Use Compose"),
]


async def test_release_gates_atomic_publication_pins_and_rollback(app, monkeypatch):
    async with client_for(app) as owner, client_for(app) as editor, client_for(app) as public:
        owner_id, org_id, root_id = await create_agent(app, owner)
        editor_id, other_org, _ = await create_agent(app, editor)
        base = f"/api/v1/organizations/{org_id}/agents"
        root_url = f"{base}/{root_id}"
        target_response = await owner.post(base, json={"name": "Delivery", "slug": "delivery"})
        assert target_response.status_code == 201
        target_id = UUID(target_response.json()["id"])
        target_url = f"{base}/{target_id}"
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            db.add(Membership(organization_id=org_id, user_id=editor_id, role="editor"))

        root, target, _, _, handoff, _ = call_flows()
        root["slots"][2]["sensitive"] = True

        async def prepare_training(agent_id, flow_config, revision, *, minimum=True):
            data = training_dataset().model_dump(mode="json")
            if minimum:
                data["examples"] = [
                    example for example in data["examples"] if example["split"] == "train"
                ]
                for intent, emoji in zip(data["intents"][:2], ("🛸", "🦄"), strict=True):
                    data["examples"].extend(
                        {
                            "id": str(uuid4()),
                            "intent_id": intent["id"],
                            "split": "test",
                            "text": emoji * repeat,
                            "expected_fallback": False,
                            "spans": [],
                        }
                        for repeat in range(1, 4)
                    )
            async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
                draft = await db.scalar(select(AgentDraft).where(AgentDraft.agent_id == agent_id))
                assert draft is not None
                draft.configuration = {
                    **draft.configuration,
                    **data,
                    "flow": flow_config["flow"],
                    "slots": flow_config["slots"],
                }
                draft.revision = revision
            return (await prepare(app, owner_id, org_id, agent_id, revision, uuid4().hex)).version

        async def build(version):
            await process_job(
                app.state.session_factory, version.job_id, org_id, HANDLERS, "training"
            )
            return version.id

        async def train(agent_id, flow_config, revision, *, minimum=True):
            return await build(
                await prepare_training(agent_id, flow_config, revision, minimum=minimum)
            )

        async def start(version_id):
            response = await owner.post(
                f"{root_url}/test-conversations",
                json={
                    "version_id": str(version_id),
                    "http_mode": "mock",
                },
            )
            assert response.status_code == 201, response.text
            return response.json()

        async def say(chat, text):
            prefix = f"/api/v1/organizations/{org_id}/test-conversations"
            endpoint = f"{prefix}/{chat['conversation_id']}/messages"
            response = await owner.post(
                endpoint,
                json={
                    "text": text,
                    "client_message_id": str(uuid4()),
                    "revision": chat["revision"],
                },
                headers={
                    "X-Conversation-Token": chat["conversation_token"],
                    "Idempotency-Key": uuid4().hex,
                },
            )
            assert response.status_code == 200, response.text
            return response.json()

        target_version = await train(target_id, target, 1)
        handoff["config"].update(
            target_agent_id=str(target_id), target_version_id=str(target_version)
        )
        short_version = await train(root_id, root, 1, minimum=False)
        rejected = await owner.get(f"{root_url}/versions/{short_version}/release-check")
        assert "release_test_minimum" in {item["code"] for item in rejected.json()["errors"]}

        first_version = await train(root_id, root, 2)
        check_url = f"{root_url}/versions/{first_version}/release-check"
        checked = (await owner.get(check_url)).json()
        assert [item["code"] for item in checked["errors"]] == ["release_smoke_required"]
        first_chat = await start(first_version)
        assert first_chat["messages"] == ["Где заказ 321?"]
        assert (await say(first_chat, "Завтра"))["status"] == "completed"
        old_chat = await start(first_version)
        etag = (await owner.get(root_url)).headers["etag"]
        publish_url = f"{root_url}/releases"
        request = {"version_id": str(first_version)}
        assert (
            await editor.post(publish_url, json=request, headers={"If-Match": etag})
        ).status_code == 403
        assert (await owner.post(publish_url, json=request)).status_code == 428
        warning = await owner.post(publish_url, json=request, headers={"If-Match": etag})
        assert warning.status_code == 409
        assert warning.json()["error"]["code"] == "release_quality_confirmation_required"
        request.update(warning_acknowledgement=True, reason="Проверено вручную")
        first, concurrent = await asyncio.gather(
            owner.post(publish_url, json=request, headers={"If-Match": etag}),
            owner.post(publish_url, json=request, headers={"If-Match": etag}),
        )
        assert sorted([first.status_code, concurrent.status_code]) == [201, 412]
        release_a = (first if first.status_code == 201 else concurrent).json()
        assert {pin["version_id"] for pin in release_a["dependency_manifest"]} == {
            str(first_version),
            str(target_version),
        }
        assert (await owner.post(f"{target_url}/archive")).status_code == 409
        widget_path = f"{root_url}/widget-channels"
        widget = await owner.post(
            widget_path,
            json={
                "name": "Поддержка",
                "allowed_origins": ["https://shop.example"],
            },
        )
        assert widget.status_code == 201
        widget_id = widget.json()["id"]
        widget_start = f"/api/v1/widget/channels/{widget_id}/sessions"
        widget_body = {"parent_origin": "https://shop.example"}
        boot_headers = {"Idempotency-Key": "start-widget"}
        boot = await public.post(widget_start, json=widget_body, headers=boot_headers)
        assert boot.status_code == 201, boot.status_code
        widget_chat = boot.json()
        assert widget_chat["messages"][0]["text"] == "Где заказ 321?"
        assert "conversation_token" not in widget_chat
        assert widget_chat["session_token"].startswith("dgw_")
        replay_boot = await public.post(widget_start, json=widget_body, headers=boot_headers)
        assert replay_boot.json()["conversation_id"] == widget_chat["conversation_id"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            bucket = await db.get(RateBucket, (org_id, f"widget:start:{widget_id}"))
            assert bucket is not None
            bucket.available_at = datetime.now(UTC) + timedelta(minutes=2)
        limited_widget = await public.post(
            widget_start, json=widget_body, headers={"Idempotency-Key": "new-session"}
        )
        assert limited_widget.status_code == 429
        assert limited_widget.headers["retry-after"] == "3"
        assert (await public.post(widget_start, json=widget_body, headers=boot_headers)).json()[
            "replayed"
        ]
        issued = await owner.post(f"{root_url}/api-keys", json={"name": "Client"})
        issued_other = await owner.post(f"{root_url}/api-keys", json={"name": "Other client"})
        assert issued.status_code == issued_other.status_code == 201
        auth = {"Authorization": f"Bearer {issued.json()['secret']}"}
        other_auth = {"Authorization": f"Bearer {issued_other.json()['secret']}"}
        runtime = "/api/v1/runtime"
        start_path = f"{runtime}/agents/{root_id}/conversations"

        async def start_public(key, credentials=auth):
            return await public.post(start_path, headers={**credentials, "Idempotency-Key": key})

        assert (await start_public("no-key", {})).status_code == 401
        assert (
            await public.post(
                f"{runtime}/agents/{target_id}/conversations",
                headers={**auth, "Idempotency-Key": "wrong-agent"},
            )
        ).status_code == 404
        # Publish a root version that no longer calls the target.
        root["flow"]["edges"][0]["target"] = root["flow"]["nodes"][3]["id"]
        root["flow"]["nodes"] = [
            node for node in root["flow"]["nodes"] if node["id"] != handoff["id"]
        ]
        root["flow"]["edges"] = [
            edge for edge in root["flow"]["edges"] if edge["source"] != handoff["id"]
        ]
        building = await prepare_training(root_id, root, 3)
        training_started, continue_training = asyncio.Event(), asyncio.Event()
        run_child = training_worker.run_child

        async def signal_training(request, settings, progress):
            async def track(stage):
                await progress(stage)
                if stage == "intent_training":
                    training_started.set()
                    await continue_training.wait()

            return await run_child(request, settings, track)

        with monkeypatch.context() as patch:
            patch.setattr(training_worker, "run_child", signal_training)
            async with asyncio.TaskGroup() as tasks:
                task = tasks.create_task(build(building))
                try:
                    await asyncio.wait_for(training_started.wait(), timeout=15)
                    during_training = (await owner.get(f"{root_url}/versions/{building.id}")).json()
                    assert during_training["job_status"] == "running"
                    public_started = await start_public("first-public")
                    assert public_started.status_code == 201, public_started.text
                    original = public_started.json()
                    assert original["release_id"] == release_a["id"]
                    assert [message["text"] for message in original["messages"]] == [
                        "Где заказ 321?"
                    ]
                    assert "trace" not in original and "revision" not in original
                finally:
                    continue_training.set()
        second_version = task.result()
        assert (await start(second_version))["status"] == "completed"
        etag = (await owner.get(root_url)).headers["etag"]
        second = await owner.post(
            publish_url,
            json={**request, "version_id": str(second_version)},
            headers={"If-Match": etag},
        )
        assert second.status_code == 201, second.text
        assert (await owner.post(f"{target_url}/archive")).status_code == 200
        assert (await say(old_chat, "Сегодня"))["messages"] == ["Доставка: Сегодня / важно"]
        widget_message = f"/api/v1/widget/conversations/{widget_chat['conversation_id']}/messages"
        widget_payload = {"client_message_id": str(uuid4()), "text": "Сегодня"}
        widget_headers = {
            "Authorization": f"Bearer {widget_chat['session_token']}",
            "Idempotency-Key": "widget-message",
        }
        crossed = f"/api/v1/widget/conversations/{uuid4()}/messages"
        assert (
            await public.post(crossed, headers=widget_headers, json=widget_payload)
        ).status_code == 401
        assert (
            await public.post(
                widget_message, headers={**widget_headers, **auth}, json=widget_payload
            )
        ).status_code == 401
        widget_answer = await public.post(
            widget_message, headers=widget_headers, json=widget_payload
        )
        assert widget_answer.status_code == 200, widget_answer.text
        assert widget_answer.json()["messages"][0]["text"] == "Доставка: Сегодня / важно"
        assert (await owner.post(f"{widget_path}/{widget_id}/disable")).status_code == 204
        assert (
            await public.post(widget_message, headers=widget_headers, json=widget_payload)
        ).status_code == 404
        assert (
            await public.post(widget_start, json=widget_body, headers=boot_headers)
        ).status_code == 403
        replacement = await start_public("replacement")
        assert replacement.status_code == 201, replacement.text
        assert replacement.json()["release_id"] == second.json()["id"]
        retried_start = (await start_public("first-public")).json()
        assert retried_start["replayed"] and retried_start["release_id"] == release_a["id"]
        assert retried_start["conversation_token"] == original["conversation_token"]

        conversation_path = f"{runtime}/conversations/{original['conversation_id']}"
        token = {"X-Conversation-Token": original["conversation_token"]}
        message = {"client_message_id": str(uuid4()), "text": "Сегодня"}
        headers = {**auth, **token, "Idempotency-Key": "message-a"}
        assert (
            await public.post(
                f"{conversation_path}/messages", json=message, headers={**headers, **other_auth}
            )
        ).status_code == 404
        assert (
            await public.post(
                f"{conversation_path}/messages",
                json=message,
                headers={
                    **headers,
                    "X-Conversation-Token": replacement.json()["conversation_token"],
                },
            )
        ).status_code == 404
        sent = await public.post(f"{conversation_path}/messages", json=message, headers=headers)
        assert sent.status_code == 200, sent.text
        result = sent.json()
        assert result["sequence"] == 1 and "trace" not in result
        assert [item["text"] for item in result["messages"]] == ["Доставка: Сегодня / важно"]
        repeat = await public.post(f"{conversation_path}/messages", json=message, headers=headers)
        assert repeat.json()["replayed"]
        assert repeat.json()["messages"] == [
            {**result["messages"][0], "text": "Доставка: Сегодня / [скрыто]"}
        ]
        assert (
            await public.post(
                f"{conversation_path}/messages", json={**message, "text": "Завтра"}, headers=headers
            )
        ).status_code == 409
        receipt_path = f"{conversation_path}/receipt/{message['client_message_id']}"
        receipt = await public.get(receipt_path, headers={**auth, **token})
        assert receipt.json() == {
            "turn_id": result["turn_id"],
            "sequence": 1,
            "status": "completed",
        }
        assert (await public.post(f"{conversation_path}/close", headers=headers)).status_code == 204
        undo = {"release_id": release_a["id"], "reason": "Возвращаем проверенный сценарий"}
        assert (
            await owner.post(
                f"{root_url}/rollback", json=undo, headers={"If-Match": second.headers["etag"]}
            )
        ).status_code == 409
        assert (await owner.post(f"{target_url}/restore")).status_code == 200
        rolled = await owner.post(
            f"{root_url}/rollback", json=undo, headers={"If-Match": second.headers["etag"]}
        )
        assert rolled.status_code == 200 and rolled.json()["current_release_id"] == release_a["id"]
        history_path = f"/api/v1/organizations/{org_id}/conversations"
        query = {"agent_id": str(root_id), "channel_type": "widget", "status": "completed"}
        stored = await editor.get(history_path, params=query)
        assert stored.status_code == 200
        assert [row["id"] for row in stored.json()["items"]] == [widget_chat["conversation_id"]]
        widget_history = f"{history_path}/{widget_chat['conversation_id']}"
        assert (await owner.get(widget_history)).json()["release_id"] == release_a["id"]
        assert (await public.get(widget_history, headers=widget_headers)).status_code == 401
        page = (await owner.get(f"{widget_history}/messages", params={"limit": 1})).json()
        assert page["next_cursor"] == "0"
        after = (
            await owner.get(f"{widget_history}/messages", params={"cursor": page["next_cursor"]})
        ).json()
        assert after["items"][0]["messages"] == [
            {"role": "user", "text": "Сегодня"},
            {"role": "agent", "text": "Доставка: Сегодня / [скрыто]"},
        ]
        trace = (await owner.get(f"{widget_history}/trace")).json()["items"]
        assert {event["version_id"] for item in trace for event in item["trace"]} == {
            str(first_version),
            str(target_version),
        }
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            counters = (
                await db.scalars(
                    select(MetricHour).where(
                        MetricHour.channel.in_(("api", "widget")),
                        MetricHour.metric.in_(("dialogues", "messages")),
                    )
                )
            ).all()
            totals = {}
            for row in counters:
                key = (row.channel, row.metric)
                totals[key] = totals.get(key, 0) + row.count
            assert totals == {
                ("api", "dialogues"): 2,
                ("api", "messages"): 1,
                ("widget", "dialogues"): 1,
                ("widget", "messages"): 1,
            }
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert (
                await db.get(ConversationHistory, (org_id, UUID(widget_chat["conversation_id"])))
                is None
            )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await db.execute(
                delete(RuntimeReceipt).where(
                    RuntimeReceipt.conversation_id == UUID(widget_chat["conversation_id"])
                )
            )
        assert (await owner.get(f"{widget_history}/messages")).json()["items"][-1][
            "messages"
        ] == after["items"][0]["messages"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            org = await db.get(Organization, org_id)
            assert org is not None
            org.store_message_text = False
        private = (await start_public("after-rollback")).json()
        assert private["release_id"] == release_a["id"]
        assert [item["text"] for item in private["messages"]] == ["Где заказ 321?"]
        assert (await start_public("after-rollback")).json()["messages"] == []
        private_path = f"{runtime}/conversations/{private['conversation_id']}"
        private_headers = {**headers, "X-Conversation-Token": private["conversation_token"]}
        delivered = await public.post(
            f"{private_path}/messages", json=message, headers=private_headers
        )
        assert delivered.status_code == 200
        assert delivered.json()["messages"][0]["text"] == "Доставка: Сегодня / важно"
        privacy_replay = (
            await public.post(f"{private_path}/messages", json=message, headers=private_headers)
        ).json()
        assert privacy_replay["replayed"] and not privacy_replay["replay_available"]
        assert privacy_replay["messages"] == []
        private_history = f"{history_path}/{private['conversation_id']}"
        assert (await owner.get(private_history)).json()["store_message_text"] is False
        assert (await owner.get(f"{private_history}/messages")).json()["items"] == []
        assert (await owner.get(f"{private_history}/trace")).json()["items"]
        org_path = f"/api/v1/organizations/{org_id}"
        etag = (await owner.get(org_path)).headers["etag"]
        assert (
            await owner.patch(
                org_path, json={"purge_existing_text": True}, headers={"If-Match": etag}
            )
        ).status_code == 200
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert (
                await db.scalar(
                    select(HistoryTurn).where(HistoryTurn.encrypted_messages.is_not(None))
                )
                is None
            )
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            bucket = await db.get(RateBucket, (org_id, f"start:key:{issued.json()['id']}"))
            assert bucket is not None
            bucket.available_at = datetime.now(UTC) + timedelta(seconds=30)
        limited = await start_public("limited")
        assert limited.status_code == 429, limited.text
        assert limited.headers["retry-after"] == "1"
        assert (await start_public("first-public")).json()["replayed"]
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            bucket = await db.get(RateBucket, (org_id, "start:organization"))
            assert bucket is not None
            bucket.available_at = datetime.now(UTC) + timedelta(seconds=30)
        assert (await start_public("limited-other-key", other_auth)).status_code == 429
        assert (
            await owner.post(f"{root_url}/api-keys/{issued.json()['id']}/revoke")
        ).status_code == 204
        assert (await start_public("revoked")).status_code == 401
        assert (await public.get(receipt_path, headers={**auth, **token})).status_code == 401
        assert (await public.post(f"{conversation_path}/close", headers=headers)).status_code == 401
        assert (
            await public.post(f"{conversation_path}/messages", json=message, headers=headers)
        ).status_code == 401
        assert (await owner.get(root_url)).json()["current_release_id"] == release_a["id"]
        history = (await owner.get(publish_url)).json()["items"]
        assert history == [second.json(), release_a]
        assert [
            agent["id"] for agent in (await owner.get(f"{base}?published=true")).json()["items"]
        ] == [str(root_id)]
        async with unit_of_work(app.state.session_factory, organization_id=other_org) as db:
            assert await db.get(Release, UUID(release_a["id"])) is None
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            audit = (
                await db.scalars(
                    select(AuditEvent).where(
                        AuditEvent.action.in_(("agent.published", "agent.rolled_back"))
                    )
                )
            ).all()
            assert len(audit) == 3 and any(
                event.details["reason"] == undo["reason"] for event in audit
            )
            published = await db.get(Release, UUID(release_a["id"]))
            assert published is not None
            with pytest.raises(DBAPIError):
                async with db.begin_nested():
                    published.reason = "Попытка изменить публикацию"
                    await db.flush()
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            await purge_tenant(db, org_id)
        async with unit_of_work(app.state.session_factory, organization_id=org_id) as db:
            assert await db.scalar(select(Release.id)) is None
            assert await db.scalar(select(ReleaseVersion.version_id)) is None
