import json
from typing import TypedDict
from uuid import UUID, uuid4

import httpx2
import pytest
from openai import AsyncOpenAI

from dialogentis.agents import empty_config
from dialogentis.config import Settings
from dialogentis.nlu import llm
from dialogentis.nlu.dialogue import DialogueChoice, DialogueTurn
from dialogentis.nlu.model_store import ModelIdentity, pack_dialogue_manifest
from dialogentis.runtime.executor import RuntimeExecutor
from dialogentis.runtime.history_context import ContextMessage, HistoryContext
from dialogentis.versions.prepare import canonical_hash
from tests.dialogue_fixtures import request_data
from tests.test_runtime_core import edge, node


class RuntimeArgs(TypedDict):
    identity: ModelIdentity
    snapshot: dict
    config_hash: str
    model: bytes
    settings: Settings


@pytest.fixture
def llm_transport(monkeypatch):
    bodies = []
    script = []

    def complete(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        bodies.append(body)
        decision = script.pop(0)
        if isinstance(decision, int):
            return httpx2.Response(decision, json={"error": {"message": "provider-only"}})
        return httpx2.Response(
            200,
            json={
                "id": "fixture-turn",
                "object": "chat.completion",
                "model": body["model"],
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(decision)},
                    }
                ],
            },
        )

    def client(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(complete))
        )

    llm._agent.cache_clear()
    monkeypatch.setattr(llm, "AsyncOpenAI", client)
    yield bodies, script
    llm._agent.cache_clear()


async def test_deepseek_uses_json_mode_and_validates_choices(llm_transport):
    requests, script = llm_transport
    script.append({"action": "route", "intent": "unavailable", "values": []})
    settings = Settings.model_validate(
        {
            "llm_base_url": "https://api.deepseek.com/v1",
            "llm_model": "deepseek-flash",
            "llm_api_key": "test-key",
        }
    )
    turn = DialogueTurn(
        version_id=uuid4(),
        node_id=uuid4(),
        text="Где моя посылка?",
        choices=(DialogueChoice(code="order", name="Статус заказа", description="Проверка"),),
        fields=(),
    )

    assert (await llm.interpret(settings, "deepseek-flash", turn)).action == "clarify"
    assert len(requests) == 1
    body = requests[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 1024 and "max_completion_tokens" not in body
    assert body["thinking"] == {"type": "disabled"} and not body.get("tools")


async def test_language_model_can_switch_branch_and_extract_grounded_slot(llm_transport, caplog):
    caplog.set_level("DEBUG")
    requests, script = llm_transport
    script.extend(
        [
            {"action": "route", "intent": "check_order", "values": []},
            {
                "action": "route",
                "intent": "cancel_order",
                "values": [{"code": "number", "value": "42", "source": "сорок два"}],
            },
        ]
    )
    identity, _, settings = request_data()
    settings = settings.model_validate(
        {
            **settings.model_dump(),
            "llm_base_url": "http://llm.invalid/v1",
            "llm_model": "fixture",
            "llm_api_key": "test-key",
        }
    )
    config = empty_config()
    _, router, _, end = config["flow"]["nodes"]
    check, cancel, slot = (str(uuid4()) for _ in range(3))
    config["intents"] = [
        {"id": check, "code": "check_order", "name": "Статус заказа"},
        {"id": cancel, "code": "cancel_order", "name": "Отмена заказа"},
    ]
    config["slots"] = [
        {"id": slot, "code": "number", "value_type": "integer"},
        {"id": str(uuid4()), "code": "private_ref", "sensitive": True, "default": "secret-ref"},
    ]
    ask_status = node("ask_slot", slot_id=slot, prompt="Какой номер заказа?")
    ask_cancel = node("ask_slot", slot_id=slot, prompt="Номер отменяемого заказа?")
    status = node("say", text="Проверяю заказ {{number}}.")
    cancellation = node("say", text="Запрос на отмену заказа {{number}}.")
    config["flow"]["nodes"] += [ask_status, ask_cancel, status, cancellation]
    config["flow"]["edges"] += [
        edge(router, ask_status, check),
        edge(router, ask_cancel, cancel),
        edge(ask_status, status, "success"),
        edge(ask_cancel, cancellation, "success"),
        edge(ask_status, ask_status, "invalid"),
        edge(ask_cancel, ask_cancel, "invalid"),
        edge(ask_status, end, "exhausted"),
        edge(ask_cancel, end, "exhausted"),
        edge(status, end),
        edge(cancellation, end),
    ]
    snapshot = {"schema_version": 1, "config": config, "llm_model": "fixture"}
    config_hash = canonical_hash(snapshot)
    args: RuntimeArgs = {
        "identity": identity,
        "snapshot": snapshot,
        "config_hash": config_hash,
        "model": pack_dialogue_manifest(identity, config_hash, settings),
        "settings": settings,
    }

    async def history():
        return HistoryContext(messages=(ContextMessage(role="agent", text="Здравствуйте!"),))

    runtime = RuntimeExecutor(timeout=20)
    try:
        first = await runtime.execute(action="start", **args)
        assert not requests
        ask = await runtime.execute(
            action="send",
            state=first.state,
            text="Посмотрите, где моя покупка secret-ref",
            history_loader=history,
            **args,
        )
        assert ask.messages == ("Какой номер заказа?",)
        switched = await runtime.execute(
            action="send",
            state=ask.state,
            text="Нет, лучше отмените сорок два",
            history_loader=history,
            **args,
        )
        assert switched.messages == ("Запрос на отмену заказа 42.",)
        assert switched.state.status == "completed"
        assert switched.nlu_observations[-1].intent_id == UUID(cancel)

        for body in requests:
            assert body["model"] == "fixture" and not body.get("tools")
            schema = body["response_format"]["json_schema"]
            assert schema["strict"]
            assert schema["schema"]["properties"]["values"]["type"] == "array"
            prompt = json.loads(body["messages"][-1]["content"])
            assert prompt["history"]["messages"] == [{"role": "agent", "text": "Здравствуйте!"}]
        assert json.loads(requests[1]["messages"][-1]["content"])["awaiting"] == "number"
        prompt = json.loads(requests[0]["messages"][-1]["content"])
        assert prompt["text"].endswith("[скрыто]") and "input_hash" not in prompt

        script.append({"action": "route", "intent": "unauthorized", "values": []})
        invalid = await runtime.execute(
            action="send", state=first.state, text="Переключи ветку", **args
        )
        assert any(item.event == "fallback" for item in invalid.trace)
        assert all(item.intent_id is None for item in invalid.nlu_observations)

        script.append(503)
        failed = await runtime.execute(
            action="send", state=first.state, text="Личное сообщение", **args
        )
        assert failed.error_code == "nlu_error"
        assert "provider-only" not in repr(failed)
        assert "Личное сообщение" not in repr(failed)
        assert "сорок два" not in caplog.text and "provider-only" not in caplog.text
    finally:
        await runtime.close()
