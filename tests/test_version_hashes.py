from copy import deepcopy

from dialogentis.versions.prepare import canonical_hash, nlu_hash
from tests.dialogue_fixtures import request_data


def test_nlu_hash_ignores_graph_and_test_set_but_tracks_model_and_examples():
    _, snapshot, _ = request_data()
    original = nlu_hash(snapshot)
    changed = deepcopy(snapshot)
    config = changed["config"]
    config["flow"]["nodes"][0]["position"]["x"] += 50
    next(item for item in config["examples"] if item["split"] == "test")["text"] += " новое"
    assert nlu_hash(changed) == original
    assert canonical_hash(changed) != canonical_hash(snapshot)
    next(item for item in config["examples"] if item["split"] == "train")["text"] += " новое"
    assert nlu_hash(changed) != original
    assert nlu_hash({**snapshot, "llm_model": "another"}) != original
