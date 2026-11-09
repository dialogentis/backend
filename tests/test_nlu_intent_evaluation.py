from uuid import uuid4

import pytest

from dialogentis.datasets.schema import Example
from dialogentis.nlu.intent import IntentPrediction, RankedIntent, train_intents
from dialogentis.nlu.intent_evaluation import evaluate_intents
from tests.test_nlu_intents import training_dataset


def test_metrics_include_unknown_predictions_and_fallbacks(monkeypatch):
    data = training_dataset()
    model = train_intents(data)
    first, second, disabled = (item.id for item in data.intents)
    data.examples = [item for item in data.examples if item.split == "train"]
    data.examples.extend(
        Example(
            id=uuid4(), text=text, intent_id=intent, expected_fallback=intent is None, split="test"
        )
        for text, intent in [
            ("known correct", first),
            ("known fallback", first),
            ("known wrong", second),
            ("unknown wrong", None),
            ("disabled ignored", disabled),
        ]
    )

    def predict(text, thresholds):
        predicted = {"known correct": first, "known wrong": first, "unknown wrong": second}.get(
            text
        )
        ranking = (RankedIntent(predicted or first, 0.8), RankedIntent(second, 0.2))
        return IntentPrediction(ranking, 0.5, () if predicted else ("low_similarity",))

    monkeypatch.setattr(model, "predict", predict)
    report = evaluate_intents(model, data)
    assert report.labels == (first, second, None)
    assert report.confusion_matrix == ((1, 0, 1), (1, 0, 0), (0, 1, 0))
    assert (report.test_count, report.known_count, report.unknown_count) == (4, 3, 1)
    assert report.excluded_disabled_count == 1
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.macro_f1 == pytest.approx(0.25)
    assert report.micro_f1 == pytest.approx(1 / 3)
    assert report.known_fallback_rate == pytest.approx(1 / 3)
    assert report.unknown_rejection == 0
    assert len(report.errors) == 3
    assert all(not hasattr(error, "text") for error in report.errors)


def test_empty_test_set_is_not_reported_as_zero_and_training_is_bound_to_model():
    data = training_dataset()
    model = train_intents(data)
    data.examples = [item for item in data.examples if item.split == "train"]
    report = evaluate_intents(model, data)
    assert report.test_count == 0
    assert report.accuracy is report.macro_f1 is report.micro_f1 is None
    assert report.unknown_rejection is report.known_fallback_rate is None
    data.examples[0].text += " изменено"
    with pytest.raises(ValueError, match="does not match"):
        evaluate_intents(model, data)


def test_real_classifier_evaluates_only_held_out_examples():
    data = training_dataset()
    model = train_intents(data)
    report = evaluate_intents(model, data)
    assert report.test_count == 2
    assert report.known_count == 1
    assert report.unknown_count == 1
    assert sum(map(sum, report.confusion_matrix)) == 2
