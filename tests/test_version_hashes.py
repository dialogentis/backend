from copy import deepcopy
from dataclasses import asdict

from dialogentis.agents import empty_config
from dialogentis.nlu.intent import Thresholds
from dialogentis.versions.prepare import canonical_hash, nlu_hash
from tests.test_nlu_intents import training_dataset


def test_nlu_hash_ignores_graph_and_test_set_but_tracks_training_and_thresholds():
    config = empty_config()
    config.update(training_dataset().model_dump(mode="json"))
    settings = asdict(Thresholds())
    original = nlu_hash(config, settings)
    changed = deepcopy(config)
    changed["flow"]["nodes"][0]["position"]["x"] += 50
    next(item for item in changed["examples"] if item["split"] == "test")["text"] += " новое"
    changed["intents"].reverse()
    changed["examples"].reverse()
    assert nlu_hash(changed, settings) == original
    assert canonical_hash(changed) != canonical_hash(config)
    next(item for item in changed["examples"] if item["split"] == "train")["text"] += " новое"
    assert nlu_hash(changed, settings) != original
    assert nlu_hash(config, {**settings, "score": 0.8}) != original
