from uuid import uuid4

import pytest

from dialogentis.datasets.schema import Span
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds
from tests.test_nlu_learned_entities import learned_dataset


def test_adapter_predicts_and_evaluates_real_intents_and_entities():
    data, city = learned_dataset()
    test_example = next(item for item in data.examples if item.split == "test" and item.intent_id)
    test_example.text = "👩🏽‍💻 доставьте в Москва номер 8"
    start = test_example.text.index("Москва")
    test_example.spans = [Span(entity_id=city, start=start, end=start + len("Москва"))]
    version_id = uuid4()
    adapter = NLUAdapter.train(version_id, data, Thresholds())
    prediction = adapter.predict(test_example.text)
    assert prediction.version_id == version_id
    assert prediction.intent.ranking
    assert [(match.entity_id, match.text) for match in prediction.entities.matches] == [
        (city, "Москва")
    ]
    unknown = adapter.predict("🦄")
    assert unknown.intent.intent_id is None
    assert unknown.intent.fallback_reason == "empty_features"
    report = adapter.evaluate(data)
    assert report.version_id == version_id
    assert report.intents.test_count == report.entities.test_count == 2
    assert (report.entities.expected, report.entities.true_positive) == (1, 1)
    data.entities[0].name = "Изменённое название"
    assert adapter.entities[0].name != data.entities[0].name
    with pytest.raises(ValueError, match="entity definitions"):
        adapter.evaluate(data)
