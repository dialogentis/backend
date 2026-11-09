from uuid import UUID, uuid4

import pytest

from dialogentis.datasets.schema import Dataset
from dialogentis.nlu.intent import InvalidTrainingDataset, Thresholds, train_intents


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


def test_fit_uses_only_enabled_training_examples_and_scores_similarity():
    data = training_dataset()
    model = train_intents(data)
    vocabulary = model.vectorizer.named_transformers.word.vocabulary_
    assert "секретноетестовоеслово" not in vocabulary
    assert "выключенноеслово" not in vocabulary
    assert model.manifest["training_examples"] == 10
    assert sum(len(part.vocabulary_) for _, part in model.vectorizer.transformer_list) <= 50000
    prediction = model.predict("где заказ 0", Thresholds(score=0, margin=0, similarity=0.99))
    assert prediction.intent_id == data.intents[0].id
    assert prediction.similarity == pytest.approx(1.0)
    assert not prediction.fallback_reasons
    assert len(prediction.ranking) == 2


def test_fallback_reasons_are_ordered_and_prevent_intent_selection():
    model = train_intents(training_dataset())
    prediction = model.predict("🦄", Thresholds(score=1, margin=1, similarity=1))
    assert prediction.fallback_reasons == (
        "empty_features",
        "low_score",
        "ambiguous_intent",
        "low_similarity",
    )
    assert prediction.intent_id is None
    assert prediction.fallback_reason == "empty_features"
    assert prediction.similarity == 0
    assert model.predict("🦄", Thresholds(score=0, margin=0, similarity=0)).fallback_reasons == (
        "empty_features",
    )
    assert model.predict("xyz", Thresholds(score=0, margin=0, similarity=0.2)).intent_id is None


def test_training_rejects_leakage_and_invalid_thresholds_or_input():
    data = training_dataset()
    data.examples[-1].text = data.examples[0].text.upper()
    with pytest.raises(InvalidTrainingDataset) as exc:
        train_intents(data)
    assert "duplicate_across_splits" in {issue.code for issue in exc.value.issues}
    assert data.examples[0].text not in str(exc.value)
    for value in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            Thresholds(score=value)
    model = train_intents(training_dataset())
    for text in ("a" * 2001, "\ud800"):
        with pytest.raises(ValueError):
            model.predict(text)


def test_training_is_stable_when_examples_are_reordered():
    data = training_dataset()
    model = train_intents(data)
    data.examples.reverse()
    data.examples[0].text = "другая независимая фраза"
    reordered = train_intents(data)
    assert reordered.manifest["training_data_sha256"] == model.manifest["training_data_sha256"]
    assert reordered.predict("где заказ 0") == model.predict("где заказ 0")


def test_training_accepts_symbol_only_examples():
    data = training_dataset()
    for index, item in enumerate(data.examples[:10]):
        item.text = ("!" if index < 5 else "?") * (index % 5 + 1)
    model = train_intents(data)
    assert (
        model.predict("!!!", Thresholds(score=0, margin=0, similarity=0.99)).intent_id
        == data.intents[0].id
    )
