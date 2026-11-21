"""Choose demo thresholds using only the development split."""

import json
from itertools import product

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from benchmarks.nlu.demo import DEVELOPMENT, load_demo_fixtures
from benchmarks.nlu.evaluate import build_dataset
from dialogentis.nlu.intent import Thresholds, train_intents

CANDIDATES = tuple(
    product(
        (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
        (0.05, 0.10, 0.15, 0.20),
        (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    )
)


def collect(name: str, fixture: dict, unknown: list[str]):
    dataset = build_dataset(name, fixture)
    labels = np.asarray([str(item.intent_id) for item in dataset.examples])
    expected, guessed, known_values, unknown_values = [], [], [], []
    disabled = Thresholds(0, 0, 0)

    def measure(model, text: str):
        result = model.predict(text, disabled)
        first, second, *_ = result.ranking
        return str(first.intent_id), (first.score, first.score - second.score, result.similarity)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in folds.split(dataset.examples, labels):
        model = train_intents(
            dataset.model_copy(update={"examples": [dataset.examples[idx] for idx in train]})
        )

        for idx in valid:
            prediction, values = measure(model, dataset.examples[idx].text)
            expected.append(labels[idx])
            guessed.append(prediction)
            known_values.append(values)
        unknown_values.extend(measure(model, text)[1] for text in unknown)
    return expected, guessed, np.asarray(known_values), np.asarray(unknown_values)


def main() -> None:
    development = json.loads(DEVELOPMENT.read_text())
    samples = {
        name: collect(name, fixture, development[name]["calibration_unknown"])
        for name, fixture in load_demo_fixtures(training_only=True).items()
    }

    def evaluate(thresholds):
        result = {}
        for name, (expected, guessed, known, unknown) in samples.items():
            accepted = np.all(known >= thresholds, axis=1)
            result[name] = {
                "intent_macro_f1": f1_score(
                    expected,
                    np.where(accepted, guessed, ""),
                    labels=sorted(set(expected)),
                    average="macro",
                ),
                "unknown_rejection": float(1 - np.all(unknown >= thresholds, axis=1).mean()),
            }
        return result

    def rank(item):
        thresholds, scores = item
        rejection = min(score["unknown_rejection"] for score in scores.values())
        f1 = min(score["intent_macro_f1"] for score in scores.values())
        return rejection >= 0.8, f1, rejection, thresholds

    chosen, results = max(((values, evaluate(values)) for values in CANDIDATES), key=rank)
    print(json.dumps({"thresholds": chosen, "cross_validation": results}, indent=2))


if __name__ == "__main__":
    main()
