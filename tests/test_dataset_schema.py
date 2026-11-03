from uuid import uuid4

from dialogentis.datasets.routes import validate_configuration
from dialogentis.datasets.schema import Dataset, validate_dataset


def example(intent_id, text, *, split="train", spans=None):
    return {
        "id": uuid4(),
        "text": text,
        "intent_id": intent_id,
        "split": split,
        "spans": spans or [],
    }


def test_valid_dataset_and_minimum_training_warnings():
    first, second = uuid4(), uuid4()
    data = Dataset.model_validate(
        {
            "intents": [
                {"id": first, "code": "first", "name": "Первое"},
                {"id": second, "code": "second", "name": "Второе"},
            ],
            "entities": [],
            "examples": [
                example(intent, f"пример {number} {intent}")
                for intent in (first, second)
                for number in range(5)
            ],
        }
    )
    errors, warnings = validate_dataset(data)
    assert errors == []
    assert [issue.code for issue in warnings] == ["few_intent_examples", "few_intent_examples"]


def test_dataset_reports_duplicate_phrases_invalid_references_and_spans():
    intent, entity = uuid4(), uuid4()
    data = Dataset.model_validate(
        {
            "intents": [{"id": intent, "code": "status", "name": "Статус"}],
            "entities": [
                {
                    "id": entity,
                    "code": "city",
                    "name": "Город",
                    "value_type": "string",
                    "method": "learned",
                }
            ],
            "examples": [
                example(
                    intent,
                    "Москва",
                    spans=[{"entity_id": entity, "start": 1, "end": 6, "text": "осква"}],
                ),
                example(
                    uuid4(),
                    "МОСКВА",
                    split="test",
                    spans=[{"entity_id": uuid4(), "start": 0, "end": 3}],
                ),
            ],
        }
    )
    errors, _ = validate_dataset(data)
    assert {
        "duplicate_across_splits",
        "unknown_intent",
        "unknown_entity",
        "token_boundary",
        "insufficient_intents",
        "insufficient_entity_examples",
    } <= {issue.code for issue in errors}


def test_regex_uses_re2_and_validates_named_groups_and_dictionary_duplicates():
    data = Dataset.model_validate(
        {
            "intents": [],
            "entities": [
                {
                    "id": uuid4(),
                    "code": "unsafe",
                    "name": "Опасное",
                    "value_type": "string",
                    "method": "regex",
                    "pattern": r"(a)\1",
                },
                {
                    "id": uuid4(),
                    "code": "number",
                    "name": "Номер",
                    "value_type": "string",
                    "method": "regex",
                    "pattern": r"(?P<id>\d+)",
                    "group": "missing",
                },
                {
                    "id": uuid4(),
                    "code": "city",
                    "name": "Город",
                    "value_type": "string",
                    "method": "dictionary",
                    "dictionary": [{"value": "Москва", "synonyms": ["москва"]}],
                },
            ],
            "examples": [],
        }
    )
    errors, _ = validate_dataset(data)
    assert {"invalid_regex", "unknown_group", "duplicate_term"} <= {issue.code for issue in errors}


def test_invalid_unicode_is_reported_without_echoing_the_value():
    errors, _ = validate_configuration(
        {
            "intents": [],
            "entities": [
                {
                    "id": str(uuid4()),
                    "code": "bad_unicode",
                    "name": "Имя",
                    "value_type": "string",
                    "method": "regex",
                    "pattern": "\ud800",
                }
            ],
            "examples": [],
        }
    )
    assert len(errors) == 1 and errors[0].code == "string_unicode"
    assert "\ud800" not in errors[0].message
