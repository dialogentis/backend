import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmarks.nlu.evaluate import build_dataset
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
