"""Evaluate fixed, held-out fixtures without tuning thresholds on the test set."""

import hashlib
import json
import platform
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds

FIXTURES = Path(__file__).with_name("fixtures.json")


def identifier(*parts: str):
    return uuid5(NAMESPACE_URL, "dialogentis/benchmark/" + "/".join(parts))


def build_dataset(name: str, fixture: dict) -> Dataset:
    entity = fixture["entity"]
    entity_id = identifier(name, "entity", entity["code"])
    intents = [
        {"id": identifier(name, "intent", item["code"]), "code": item["code"], "name": item["name"]}
        for item in fixture["intents"]
    ]
    examples = []
    for item in fixture["intents"]:
        for split in ("train", "test"):
            for text in item[split]:
                spans = []
                for term in entity["values"]:
                    for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I):
                        spans.append(
                            {"entity_id": entity_id, "start": match.start(), "end": match.end()}
                        )
                examples.append(
                    {
                        "id": identifier(name, split, text),
                        "text": text,
                        "intent_id": identifier(name, "intent", item["code"]),
                        "split": split,
                        "spans": spans,
                    }
                )
    for text in fixture["unknown"]:
        examples.append(
            {
                "id": identifier(name, "unknown", text),
                "text": text,
                "expected_fallback": True,
                "split": "test",
                "spans": [],
            }
        )
    return Dataset.model_validate(
        {
            "intents": intents,
            "entities": [
                {
                    "id": entity_id,
                    "code": entity["code"],
                    "name": entity["name"],
                    "method": "dictionary",
                    "value_type": "string",
                    "dictionary": [{"value": term} for term in entity["values"]],
                }
            ],
            "examples": examples,
        }
    )


def evaluate() -> dict:
    content = FIXTURES.read_bytes()
    fixtures = json.loads(content)
    results = {}
    for name, fixture in fixtures.items():
        dataset = build_dataset(name, fixture)
        errors, warnings = validate_dataset(dataset)
        if errors:
            raise ValueError(
                f"{name}: invalid dataset: {[(item.path, item.code) for item in errors]}"
            )
        counts = Counter(example.split for example in dataset.examples)
        model = NLUAdapter.train(identifier(name, "version"), dataset, Thresholds())
        report = model.evaluate(dataset)
        results[name] = {
            "name": fixture["name"],
            "intents": len(dataset.intents),
            "train": counts["train"],
            "test": report.intents.known_count,
            "unknown": report.intents.unknown_count,
            "intent_macro_f1": report.intents.macro_f1,
            "entity_f1": report.entities.f1,
            "entity_expected_spans": report.entities.expected,
            "unknown_rejection": report.intents.unknown_rejection,
            "known_fallback_rate": report.intents.known_fallback_rate,
            "warnings": [item.code for item in warnings],
        }
    return {
        "fixture_sha256": hashlib.sha256(content).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(
            (FIXTURES.parents[2] / "uv.lock").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            name: version(name) for name in ("scikit-learn", "python-crfsuite", "numpy", "scipy")
        },
        "thresholds": {"score": 0.55, "margin": 0.15, "similarity": 0.20},
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2, allow_nan=False))
