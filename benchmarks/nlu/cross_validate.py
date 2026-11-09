"""Compare regularization using only training examples, never held-out benchmark phrases."""

import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from benchmarks.nlu.evaluate import FIXTURES, build_dataset
from dialogentis.nlu.intent import Thresholds, train_intents, training_examples

CANDIDATES = (1.0, 3.0, 10.0, 30.0, 100.0)


def calculate(fixture_name: str, fixture: dict) -> dict:
    training_fixture = {
        **fixture,
        "intents": [{**item, "test": []} for item in fixture["intents"]],
        "unknown": [],
    }
    dataset = build_dataset(fixture_name, training_fixture)
    examples = dataset.examples
    labels = [str(item.intent_id) for item in examples]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    counts = {candidate: defaultdict(int) for candidate in CANDIDATES}

    def run(train_data, validation, known):
        model = train_intents(train_data)
        train_labels = [str(item.intent_id) for item in training_examples(train_data)]
        for candidate in CANDIDATES:
            model.classifier.set_params(C=candidate)
            model.classifier.fit(model.train_vectors, train_labels)
            for item in validation:
                prediction = model.predict(item.text, Thresholds())
                if known:
                    counts[candidate]["known_total"] += 1
                    counts[candidate]["known_correct"] += prediction.intent_id == item.intent_id
                    counts[candidate]["known_wrong"] += prediction.intent_id not in (
                        None,
                        item.intent_id,
                    )
                    counts[candidate]["known_fallback"] += prediction.intent_id is None
                else:
                    counts[candidate]["unknown_total"] += 1
                    counts[candidate]["unknown_rejected"] += prediction.intent_id is None

    for train_indices, valid_indices in splitter.split(examples, labels):
        train = dataset.model_copy(deep=True)
        train.examples = [examples[i].model_copy(deep=True) for i in train_indices]
        run(train, [examples[i] for i in valid_indices], True)
    for intent in dataset.intents:
        train = dataset.model_copy(deep=True)
        train.intents = [
            item.model_copy(update={"enabled": item.id != intent.id}) for item in train.intents
        ]
        train.examples = [item for item in train.examples if item.intent_id != intent.id]
        run(train, [item for item in examples if item.intent_id == intent.id], False)
    return {str(candidate): dict(counts[candidate]) for candidate in CANDIDATES}


def main():
    fixtures = json.loads(Path(FIXTURES).read_text())
    print(
        json.dumps(
            {
                "method": (
                    "5-fold stratified known + leave-one-intent-out unknown proxy; "
                    "training split only"
                ),
                "seed": 42,
                "thresholds": {"score": 0.55, "margin": 0.15, "similarity": 0.2},
                "results": {name: calculate(name, fixture) for name, fixture in fixtures.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
