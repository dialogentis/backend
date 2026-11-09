from uuid import UUID

import pytest

from dialogentis.datasets.preview import preview_entity
from dialogentis.datasets.schema import Entity
from dialogentis.nlu.entities import extract_entities
from dialogentis.nlu.learned_entities import train_learned_entities
from tests.test_nlu_learned_entities import learned_dataset


def make_entity(number, code, method, **values):
    return Entity(
        id=UUID(int=number), code=code, name=code, value_type="string", method=method, **values
    )


def test_regex_precedes_dictionary_and_shorter_nonoverlapping_match_survives():
    regex = make_entity(2, "word", "regex", pattern="город")
    dictionary = make_entity(
        1,
        "city",
        "dictionary",
        dictionary=[{"value": "регион", "synonyms": ["город Москва", "Москва"]}],
    )
    text = "город Москва"
    preview = preview_entity(dictionary, text)
    assert [(match.text, match.value) for match in preview.matches] == [(text, "регион")]
    result = extract_entities([dictionary, regex], None, text)
    assert [(match.source, match.text, match.value) for match in result.matches] == [
        ("regex", "город", "город"),
        ("dictionary", "Москва", "регион"),
    ]
    assert [(item.candidate.text, item.overlaps.text) for item in result.rejected] == [
        (text, "город")
    ]


def test_dictionary_longest_and_uuid_ties_do_not_depend_on_input_order():
    short = make_entity(1, "short", "dictionary", dictionary=[{"value": "Москва"}])
    long = make_entity(2, "long", "dictionary", dictionary=[{"value": "город Москва"}])
    same = make_entity(3, "same", "dictionary", dictionary=[{"value": "город Москва"}])
    for entities in ([short, same, long], [long, same, short]):
        result = extract_entities(entities, None, "город Москва")
        assert [(item.entity_id, item.text) for item in result.matches] == [
            (long.id, "город Москва")
        ]
        assert {item.candidate.entity_id for item in result.rejected} == {short.id, same.id}


def test_real_learned_output_loses_to_regex_and_keeps_offsets():
    data, city = learned_dataset()
    model = train_learned_entities(data)
    assert model is not None
    regex = make_entity(99, "city_regex", "regex", pattern="Москва")
    text = "👩🏽‍💻 доставьте в Москва номер 8"
    result = extract_entities([*data.entities, regex], model, text)
    assert len(result.matches) == 1 and result.matches[0].source == "regex"
    assert len(result.rejected) == 1 and result.rejected[0].candidate.entity_id == city
    assert (
        result.rejected[0].candidate.text == text[result.matches[0].start : result.matches[0].end]
    )


def test_limits_and_model_definition_mismatch_are_errors(monkeypatch):
    item = make_entity(1, "letter", "regex", pattern="a")
    monkeypatch.setattr("dialogentis.nlu.entities.MAX_CANDIDATES", 2)
    with pytest.raises(ValueError, match="limit exceeded"):
        extract_entities([item], None, "aaaa")
    data, _ = learned_dataset()
    with pytest.raises(ValueError, match="does not match"):
        extract_entities(data.entities, None, "Москва")
    with pytest.raises(ValueError, match="Duplicate"):
        extract_entities([item, item], None, "a")
