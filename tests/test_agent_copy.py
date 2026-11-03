from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi import HTTPException

from dialogentis.agents import copy_config, empty_config


def test_copy_rejects_unrecognized_references_without_mutating_source():
    config = empty_config()
    config["flow"]["nodes"][2]["config"]["text"] = {"hidden_reference": "unknown"}
    with pytest.raises(HTTPException) as error:
        copy_config(config)
    assert error.value.status_code == 409
    assert config["flow"]["nodes"][2]["config"]["text"] == {"hidden_reference": "unknown"}


def test_copy_rejects_missing_resume_node():
    config = empty_config()
    config["flow"]["nodes"][2]["config"]["resume_node_id"] = ["invalid"]
    with pytest.raises(HTTPException) as error:
        copy_config(config)
    assert error.value.status_code == 409


def test_copy_rejects_unsupported_output_handles():
    config = empty_config()
    config["flow"]["edges"][1]["source_handle"] = "an-unknown-intent-id"
    with pytest.raises(HTTPException) as error:
        copy_config(config)
    assert error.value.status_code == 409


def test_copy_remaps_dataset_references_without_replacing_literal_text():
    config = empty_config()
    intent_id, entity_id, example_id = map(str, (uuid4(), uuid4(), uuid4()))
    config["intents"] = [{"id": intent_id, "code": "order", "name": "Заказ"}]
    config["entities"] = [
        {
            "id": entity_id,
            "code": "city",
            "name": "Город",
            "value_type": "string",
            "method": "learned",
        }
    ]
    config["examples"] = [
        {
            "id": example_id,
            "text": intent_id,
            "intent_id": intent_id,
            "split": "train",
            "spans": [{"entity_id": entity_id, "start": 0, "end": 1}],
        }
    ]
    original = deepcopy(config)
    clone = copy_config(config)
    assert config == original
    new_intent, new_entity, new_example = (
        clone["intents"][0],
        clone["entities"][0],
        clone["examples"][0],
    )
    assert (
        new_intent["id"] != intent_id
        and new_entity["id"] != entity_id
        and new_example["id"] != example_id
    )
    assert new_example["intent_id"] == new_intent["id"]
    assert new_example["spans"][0]["entity_id"] == new_entity["id"]
    assert new_example["text"] == intent_id


def test_copy_rejects_unknown_dataset_references():
    config = empty_config()
    config["examples"] = [
        {"id": str(uuid4()), "text": "Пример", "intent_id": str(uuid4()), "split": "train"}
    ]
    with pytest.raises(HTTPException) as error:
        copy_config(config)
    assert error.value.status_code == 409
