from uuid import UUID, uuid4

import pytest

from dialogentis.datasets.schema import Entity, Example, Span
from dialogentis.nlu.entity_evaluation import evaluate_entities
from dialogentis.nlu.learned_entities import train_learned_entities
from tests.test_nlu_intents import training_dataset
from tests.test_nlu_learned_entities import learned_dataset


def test_exact_span_metrics_and_normalization_errors():
    data = training_dataset()
    city = UUID(int=10)
    data.entities.append(
        Entity(
            id=city,
            code="city",
            name="Город",
            value_type="integer",
            method="dictionary",
            dictionary=[{"value": "Москва"}, {"value": "Казань"}],
        )
    )
    data.examples = [item for item in data.examples if item.split == "train"]
    first = data.intents[0].id
    data.examples.extend(
        [
            Example(
                id=uuid4(),
                text="город Москва",
                intent_id=first,
                split="test",
                spans=[Span(entity_id=city, start=6, end=12)],
            ),
            Example(
                id=uuid4(),
                text="город Пермь",
                intent_id=first,
                split="test",
                spans=[Span(entity_id=city, start=6, end=11)],
            ),
            Example(id=uuid4(), text="город Казань", expected_fallback=True, split="test"),
            Example(
                id=uuid4(), text="выключенный тест", intent_id=data.intents[2].id, split="test"
            ),
        ]
    )
    report = evaluate_entities(data, None)
    assert (report.test_count, report.excluded_disabled_count) == (3, 1)
    assert (report.expected, report.predicted, report.true_positive) == (2, 2, 1)
    assert report.precision == report.recall == report.f1 == 0.5
    assert report.invalid_values == 2
    assert len(report.errors) == 3
    assert report.per_entity[0].code == "city"
    assert all(not hasattr(error, "text") for error in report.errors)


def test_empty_test_set_has_no_metrics_and_changed_training_rejected():
    data, _ = learned_dataset()
    model = train_learned_entities(data)
    assert model is not None
    data.examples = [item for item in data.examples if item.split == "train"]
    report = evaluate_entities(data, model)
    assert report.test_count == 0
    assert report.precision is report.recall is report.f1 is None
    data.examples[0].text += " изменено"
    with pytest.raises(ValueError, match="does not match"):
        evaluate_entities(data, model)
