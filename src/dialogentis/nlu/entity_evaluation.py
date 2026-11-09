"""Exact-span evaluation of the combined entity extractors on held-out examples."""

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.nlu.entities import extract_entities
from dialogentis.nlu.intent import InvalidTrainingDataset
from dialogentis.nlu.intent_evaluation import ratio
from dialogentis.nlu.learned_entities import LearnedEntityModel, training_data_hash


@dataclass(frozen=True)
class EntitySpan:
    entity_id: UUID
    start: int
    end: int


@dataclass(frozen=True)
class EntityMetrics:
    entity_id: UUID
    code: str
    expected: int
    predicted: int
    true_positive: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class EntityEvaluationError:
    example_id: UUID
    missing: tuple[EntitySpan, ...]
    unexpected: tuple[EntitySpan, ...]
    invalid_values: tuple[EntitySpan, ...]


@dataclass(frozen=True)
class EntityEvaluation:
    test_count: int
    excluded_disabled_count: int
    expected: int
    predicted: int
    true_positive: int
    precision: float | None
    recall: float | None
    f1: float | None
    invalid_values: int
    per_entity: tuple[EntityMetrics, ...]
    errors: tuple[EntityEvaluationError, ...]


def span_key(span: EntitySpan) -> tuple[str, int, int]:
    return str(span.entity_id), span.start, span.end


def evaluate_entities(dataset: Dataset, model: LearnedEntityModel | None) -> EntityEvaluation:
    errors, _ = validate_dataset(dataset)
    if errors:
        raise InvalidTrainingDataset(errors)
    learned_types = {
        item.id: item.value_type for item in dataset.entities if item.method == "learned"
    }
    if (model.value_types if model else {}) != learned_types:
        raise ValueError("Learned model does not match entity definitions")
    if model and model.manifest["training_data_sha256"] != training_data_hash(dataset):
        raise ValueError("The test dataset does not match the model's training data")
    enabled = {item.id for item in dataset.intents if item.enabled}
    counts: dict[UUID, Counter[str]] = {item.id: Counter() for item in dataset.entities}
    error_items: list[EntityEvaluationError] = []
    tested = excluded = invalid_count = 0
    for example in dataset.examples:
        if example.split != "test":
            continue
        if not example.expected_fallback and example.intent_id not in enabled:
            excluded += 1
            continue
        tested += 1
        result = extract_entities(dataset.entities, model, example.text)
        expected = {EntitySpan(item.entity_id, item.start, item.end) for item in example.spans}
        predicted = {EntitySpan(item.entity_id, item.start, item.end) for item in result.matches}
        invalid = {
            EntitySpan(item.entity_id, item.start, item.end)
            for item in result.matches
            if item.error
        }
        for span in expected:
            counts[span.entity_id]["expected"] += 1
        for span in predicted:
            counts[span.entity_id]["predicted"] += 1
        for span in expected & predicted:
            counts[span.entity_id]["tp"] += 1
        invalid_count += len(invalid)
        missing, unexpected = expected - predicted, predicted - expected
        if missing or unexpected or invalid:
            error_items.append(
                EntityEvaluationError(
                    example.id,
                    tuple(sorted(missing, key=span_key)),
                    tuple(sorted(unexpected, key=span_key)),
                    tuple(sorted(invalid, key=span_key)),
                )
            )
    per_entity = []
    for item in sorted(dataset.entities, key=lambda entity: str(entity.id)):
        counter = counts[item.id]
        expected, predicted, tp = counter["expected"], counter["predicted"], counter["tp"]
        per_entity.append(
            EntityMetrics(
                item.id,
                item.code,
                expected,
                predicted,
                tp,
                ratio(tp, predicted),
                ratio(tp, expected),
                ratio(2 * tp, expected + predicted),
            )
        )
    expected = sum(item.expected for item in per_entity)
    predicted = sum(item.predicted for item in per_entity)
    tp = sum(item.true_positive for item in per_entity)
    return EntityEvaluation(
        tested,
        excluded,
        expected,
        predicted,
        tp,
        ratio(tp, predicted),
        ratio(tp, expected),
        ratio(2 * tp, expected + predicted),
        invalid_count,
        tuple(per_entity),
        tuple(error_items),
    )
