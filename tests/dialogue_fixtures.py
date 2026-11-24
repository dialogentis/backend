import json
from uuid import UUID, uuid4

import httpx2
from cryptography.fernet import Fernet
from openai import AsyncOpenAI

from dialogentis.agents import empty_config
from dialogentis.config import Settings
from dialogentis.datasets.schema import Dataset
from dialogentis.nlu import llm
from dialogentis.nlu.dialogue import DialogueDecision, DialogueTurn, DialogueValue
from dialogentis.nlu.model_store import ModelIdentity


def training_dataset() -> Dataset:
    first = UUID("00000000-0000-0000-0000-000000000001")
    second = UUID("00000000-0000-0000-0000-000000000002")
    disabled = UUID("00000000-0000-0000-0000-000000000003")
    return Dataset.model_validate(
        {
            "intents": [
                {"id": first, "code": "order", "name": "Заказ"},
                {"id": second, "code": "return", "name": "Возврат"},
                {"id": disabled, "code": "disabled", "name": "Выключено", "enabled": False},
            ],
            "entities": [],
            "examples": [
                {"id": uuid4(), "text": f"{phrase} {number}", "intent_id": intent, "split": "train"}
                for intent, phrase in [(first, "где заказ"), (second, "вернуть покупку")]
                for number in range(5)
            ]
            + [
                {
                    "id": uuid4(),
                    "text": "выключенноеслово",
                    "intent_id": disabled,
                    "split": "train",
                },
                {
                    "id": uuid4(),
                    "text": "секретноетестовоеслово",
                    "intent_id": first,
                    "split": "test",
                },
                {
                    "id": uuid4(),
                    "text": "неизвестная фраза",
                    "expected_fallback": True,
                    "split": "test",
                },
            ],
        }
    )


def request_data():
    config = empty_config()
    config.update(training_dataset().model_dump(mode="json"))
    identity = ModelIdentity(uuid4(), uuid4(), uuid4(), uuid4())
    snapshot = {"schema_version": 1, "config": config, "llm_model": "fixture"}
    return (
        identity,
        snapshot,
        Settings(
            encryption_key=Fernet.generate_key().decode(),
            llm_base_url="",
            llm_model="",
            llm_api_key=None,
        ),
    )


def clarify(_):
    return DialogueDecision(action="clarify", intent=None, values=(), reply=None)


def route(code="order", *values):
    return DialogueDecision(action="route", intent=code, values=values, reply=None)


def value(code, text, source=None):
    return DialogueValue(code=code, value=text, source=source or text)


def answer(turn):
    if not turn.awaiting:
        return clarify(turn)
    return DialogueDecision(
        action="continue", intent=None, reply=None, values=(value(turn.awaiting, turn.text),)
    )


def fixture_completion(body):
    turn = DialogueTurn.model_validate(
        {
            **json.loads(body["messages"][-1]["content"]),
            "version_id": uuid4(),
            "node_id": uuid4(),
        }
    )
    text = turn.text.casefold().strip()
    choice = next(
        (choice for choice in turn.choices if text in {e.casefold() for e in choice.examples}),
        None,
    )
    decision = route(choice.code) if choice else answer(turn)
    return {
        "id": "dialogentis-test-turn",
        "object": "chat.completion",
        "model": body["model"],
        "created": 0,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": decision.model_dump_json()},
            }
        ],
    }


def install_provider(monkeypatch):
    def complete(request):
        return httpx2.Response(200, json=fixture_completion(json.loads(request.content)))

    def client(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(complete))
        )

    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "fixture")
    llm._agent.cache_clear()
    monkeypatch.setattr(llm, "AsyncOpenAI", client)
