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
