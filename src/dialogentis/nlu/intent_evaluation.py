"""Metrics for a held-out set; this module never fits a model."""

from dataclasses import dataclass
from uuid import UUID

from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.nlu.intent import (
    FallbackReason,
    IntentModel,
    InvalidTrainingDataset,
    Thresholds,
    training_data_hash,
)


@dataclass(frozen=True)
class ClassMetrics:
    intent_id: UUID
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class EvaluationError:
    example_id: UUID
    expected: UUID | None
    predicted: UUID | None
    fallback_reasons: tuple[FallbackReason, ...]


@dataclass(frozen=True)
class IntentEvaluation:
    labels: tuple[UUID | None, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    test_count: int
    known_count: int
    unknown_count: int
    excluded_disabled_count: int
    accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    unknown_rejection: float | None
    known_fallback_rate: float | None
    per_intent: tuple[ClassMetrics, ...]
    errors: tuple[EvaluationError, ...]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def evaluate_intents(
    model: IntentModel, dataset: Dataset, thresholds: Thresholds | None = None
) -> IntentEvaluation:
    errors, _ = validate_dataset(dataset)
    if errors:
        raise InvalidTrainingDataset(errors)
    if model.manifest["training_data_sha256"] != training_data_hash(dataset):
        raise ValueError("The test dataset does not match the model's training data")
    enabled = sorted((intent.id for intent in dataset.intents if intent.enabled), key=str)
    if {str(intent) for intent in enabled} != set(model.classifier.classes_):
        raise ValueError("The test dataset does not match the model's intent classes")
    labels: tuple[UUID | None, ...] = (*enabled, None)
    indexes = {label: index for index, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    evaluation_errors: list[EvaluationError] = []
    excluded = 0
    for item in dataset.examples:
        if item.split != "test":
            continue
        expected = None if item.expected_fallback else item.intent_id
        if expected not in indexes:
            excluded += 1
            continue
        prediction = model.predict(item.text, thresholds)
        predicted = prediction.intent_id
        matrix[indexes[expected]][indexes[predicted]] += 1
        if expected != predicted:
            evaluation_errors.append(
                EvaluationError(item.id, expected, predicted, prediction.fallback_reasons)
            )

    metrics: list[ClassMetrics] = []
    for index, intent_id in enumerate(enabled):
        tp = matrix[index][index]
        support = sum(matrix[index])
        fp = sum(row[index] for row in matrix) - tp
        fn = support - tp
        metrics.append(
            ClassMetrics(
                intent_id,
                support,
                tp,
                fp,
                fn,
                ratio(tp, tp + fp),
                ratio(tp, tp + fn),
                ratio(2 * tp, 2 * tp + fp + fn),
            )
        )
    tp = sum(item.true_positive for item in metrics)
    fp = sum(item.false_positive for item in metrics)
    fn = sum(item.false_negative for item in metrics)
    known = sum(sum(row) for row in matrix[:-1])
    unknown = sum(matrix[-1])
    return IntentEvaluation(
        labels=labels,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        test_count=known + unknown,
        known_count=known,
        unknown_count=unknown,
        excluded_disabled_count=excluded,
        accuracy=ratio(tp, known),
        macro_precision=mean([item.precision for item in metrics]),
        macro_recall=mean([item.recall for item in metrics]),
        macro_f1=mean([item.f1 for item in metrics]),
        micro_precision=ratio(tp, tp + fp),
        micro_recall=ratio(tp, tp + fn),
        micro_f1=ratio(2 * tp, 2 * tp + fp + fn),
        unknown_rejection=ratio(matrix[-1][-1], unknown),
        known_fallback_rate=ratio(sum(row[-1] for row in matrix[:-1]), known),
        per_intent=tuple(metrics),
        errors=tuple(evaluation_errors),
    )
