import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from benchmarks.nlu.cross_validate import calculate
from benchmarks.nlu.evaluate import build_dataset, identifier
from dialogentis.datasets.schema import validate_dataset

FIXTURE_DIR = Path(__file__).parents[1] / "benchmarks" / "nlu"


def test_benchmark_is_fixed_disjoint_and_has_required_coverage():
    content = (FIXTURE_DIR / "fixtures.json").read_bytes()
    recorded = json.loads((FIXTURE_DIR / "baseline.json").read_text())
    assert hashlib.sha256(content).hexdigest() == recorded["fixture_sha256"]
    fixtures = json.loads(content)
    assert set(fixtures) == {"order_support", "education"}
    for name, fixture in fixtures.items():
        dataset = build_dataset(name, fixture)
        errors, _ = validate_dataset(dataset)
        assert not errors
        assert len(dataset.intents) == 5
        train = Counter(item.intent_id for item in dataset.examples if item.split == "train")
        test = Counter(item.intent_id for item in dataset.examples if item.split == "test")
        assert all(train[intent.id] >= 10 and test[intent.id] >= 5 for intent in dataset.intents)
        assert sum(item.expected_fallback for item in dataset.examples) >= 20
        assert any(item.spans for item in dataset.examples if item.split == "test")


def test_cross_validation_never_uses_held_out_examples():
    fixture = json.loads((FIXTURE_DIR / "fixtures.json").read_text())["order_support"]
    changed = deepcopy(fixture)
    for item in changed["intents"]:
        item["test"] = ["изменённая отложенная фраза"]
    changed["unknown"] = ["другой неизвестный запрос"]
    assert calculate("order_support", fixture) == calculate("order_support", changed)


def test_held_out_entity_spans_match_manual_annotation():
    reviewed = {
        "order_support": {"delivery": {1: "Казань", 2: "Пермь", 4: "Томск"}},
        "education": {
            "enrollment": {1: "Аналитика", 3: "Python", 4: "Маркетинг"},
            "schedule": {1: "Английский"},
            "course_price": {1: "Python", 2: "Дизайн", 3: "Английский"},
        },
    }
    for name, fixture in json.loads((FIXTURE_DIR / "fixtures.json").read_text()).items():
        expected = set()
        entity_id = identifier(name, "entity", fixture["entity"]["code"])
        for intent in fixture["intents"]:
            for index, mention in reviewed[name].get(intent["code"], {}).items():
                text = intent["test"][index - 1]
                start = text.index(mention)
                expected.add(
                    (identifier(name, "test", text), entity_id, start, start + len(mention))
                )
        actual = {
            (example.id, span.entity_id, span.start, span.end)
            for example in build_dataset(name, fixture).examples
            if example.split == "test"
            for span in example.spans
        }
        assert actual == expected
