"""Choose demo thresholds using only the development split."""

import json
from itertools import product

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from benchmarks.nlu.demo import DEVELOPMENT, load_demo_fixtures
from benchmarks.nlu.evaluate import build_dataset
from dialogentis.nlu.intent import Thresholds, train_intents

SCORES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
GRIDS = {
    "order_support": (
        (1.0,),
        (0.05, 0.10, 0.15, 0.20),
        (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
        (0.0,),
    ),
    "education": (
        (1.0, 3.0, 10.0),
        (0.04, 0.08, 0.12, 0.16, 0.20),
        (0.15, 0.20, 0.25, 0.30, 0.35),
        (0.0, 0.25, 0.35, 0.45, 0.55, 0.65),
    ),
}


def collect(name: str, fixture: dict, unknown: list[str], classifier_c: float):
    dataset = build_dataset(name, fixture)
    labels = np.asarray([str(item.intent_id) for item in dataset.examples])
    expected, guessed, known_values, unknown_values = [], [], [], []
    disabled = Thresholds(0, 0, 0)

    def measure(model, text: str):
        result = model.predict(text, disabled)
        first, second, *_ = result.ranking
        values = (
            first.score,
            first.score - second.score,
            result.similarity,
            model.vocabulary_coverage(text),
        )
        return str(first.intent_id), values

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in folds.split(dataset.examples, labels):
        model = train_intents(
            dataset.model_copy(update={"examples": [dataset.examples[idx] for idx in train]}),
            classifier_c=classifier_c,
        )
        for idx in valid:
            prediction, values = measure(model, dataset.examples[idx].text)
            expected.append(labels[idx])
            guessed.append(prediction)
            known_values.append(values)
        unknown_values.extend(measure(model, text)[1] for text in unknown)
    return expected, guessed, np.asarray(known_values), np.asarray(unknown_values)


def choose(name: str, fixture: dict, calibration: list[str]) -> dict:
    coefficients, margins, similarities, coverage = GRIDS[name]

    def candidates():
        for coefficient in coefficients:
            expected, guessed, known, unknown = collect(name, fixture, calibration, coefficient)
            labels = sorted(set(expected))
            for thresholds in product(SCORES, margins, similarities, coverage):
                accepted = np.all(known >= thresholds, axis=1)
                predicted = np.where(accepted, guessed, "")
                f1 = float(f1_score(expected, predicted, labels=labels, average="macro"))
                rejection = float(1 - np.all(unknown >= thresholds, axis=1).mean())
                yield (rejection >= 0.8, f1, rejection, -coefficient, thresholds)

    _, f1, rejection, negative_c, thresholds = max(candidates())
    return {
        "thresholds": thresholds,
        "classifier_c": -negative_c,
        "intent_macro_f1": f1,
        "unknown_rejection": rejection,
    }


def main() -> None:
    development = json.loads(DEVELOPMENT.read_text())
    samples = {
        name: choose(name, fixture, development[name]["calibration_unknown"])
        for name, fixture in load_demo_fixtures(training_only=True).items()
    }
    print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
