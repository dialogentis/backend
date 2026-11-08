import asyncio
from threading import Event
from uuid import uuid4

import pytest

from dialogentis.datasets.preview import Preview, convert_value, preview_entity, run_preview
from dialogentis.datasets.schema import Entity


def entity(**values):
    return Entity(id=uuid4(), code="entity", name="Сущность", value_type="string", **values)


def test_dictionary_matches_whole_tokens_longest_and_unicode_offsets():
    item = entity(
        method="dictionary",
        dictionary=[
            {"value": "центр", "synonyms": ["Москва"]},
            {"value": "регион", "synonyms": ["город Москва"]},
            {"value": "улица", "synonyms": ["Straße"]},
        ],
    )
    text = "👩🏽‍💻 город   МОСКВА, STRASSE и Подмосковье"
    result = preview_entity(item, text)
    assert [(match.text, match.value) for match in result.matches] == [
        ("город   МОСКВА", "регион"),
        ("STRASSE", "улица"),
    ]
    assert all(text[match.start : match.end] == match.text for match in result.matches)
    assert result.matches[0].start == len("👩🏽‍💻 ")


def test_regex_groups_zero_length_and_limit():
    item = entity(method="regex", pattern=r"№(?P<id>\d+)", group="id")
    result = preview_entity(item, "📦 №123 и №7")
    assert [(match.text, match.start, match.end) for match in result.matches] == [
        ("123", 3, 6),
        ("7", 10, 11),
    ]
    item.pattern = r"\d"
    item.group = 0
    assert preview_entity(item, "1" * 101).truncated
    item.pattern = r"a*"
    assert [match.text for match in preview_entity(item, "aaa").matches] == ["aaa"]
    item.pattern = r"(a)?b"
    item.group = 1
    assert [match.text for match in preview_entity(item, "b ab").matches] == ["a"]
    item.pattern = r"(a)\1"
    with pytest.raises(ValueError):
        preview_entity(item, "aa")


def test_value_conversion_is_explicit_and_never_returns_nonfinite_values():
    assert convert_value(" -42 ", "integer") == -42
    assert convert_value("да", "boolean") is True
    assert convert_value("2026-11-09", "date") == "2026-11-09"
    for value, kind in [
        ("1,2", "number"),
        ("NaN", "number"),
        ("1e-999", "number"),
        ("9007199254740992", "integer"),
        ("09.11.2026", "date"),
    ]:
        with pytest.raises(ValueError):
            convert_value(value, kind)
    item = Entity(
        id=uuid4(),
        code="amount",
        name="Сумма",
        value_type="integer",
        method="regex",
        pattern=r"\p{L}+",
    )
    match = preview_entity(item, "ошибка").matches[0]
    assert match.value is None and match.error == "invalid_value"


async def test_cancelled_request_keeps_slot_until_thread_finishes(monkeypatch):
    started, finish = Event(), Event()

    def slow_preview(entity, text):
        started.set()
        finish.wait(timeout=3)
        return Preview(matches=[], truncated=False)

    monkeypatch.setattr("dialogentis.datasets.preview.preview_entity", slow_preview)
    slots = asyncio.Semaphore(1)
    task = asyncio.create_task(run_preview(entity(method="regex", pattern="a"), "a", slots))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slots.locked()
    finish.set()
    async with asyncio.timeout(1):
        await slots.acquire()
    slots.release()
