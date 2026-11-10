import json
from copy import deepcopy
from uuid import uuid4

import pytest

from dialogentis.agents import empty_config
from dialogentis.datasets.json_transfer import PortableConfig, export_json, parse_json, preview_json


def populated():
    config = empty_config()
    intent_id, entity_id, example_id = (str(uuid4()) for _ in range(3))
    text = '=👩🏽‍💻 "заказ"\n123'
    start = text.index("123")
    config["intents"] = [{"id": intent_id, "code": "order", "name": "Заказ"}]
    config["entities"] = [
        {
            "id": entity_id,
            "code": "number",
            "name": "Номер",
            "value_type": "string",
            "method": "learned",
        }
    ]
    config["examples"] = [
        {
            "id": example_id,
            "text": text,
            "intent_id": intent_id,
            "split": "train",
            "spans": [{"entity_id": entity_id, "start": start, "end": start + 3}],
        }
    ]
    config["flow"]["nodes"][2]["config"]["text"] = "Попробуйте ещё раз"
    return PortableConfig.model_validate(config)


def encode(config):
    return json.dumps({"schema_version": 1, "config": config}, ensure_ascii=False).encode()


def test_json_round_trip_preserves_text_offsets_and_graph():
    config = populated()
    parsed, errors = parse_json(export_json(config))
    assert not errors and parsed == config
    assert parsed is not None
    preview = preview_json(parsed, PortableConfig.model_validate(empty_config()), "replace")
    assert not preview.errors and preview.config == config
    assert preview.config is not None
    example = preview.config.examples[0]
    assert example.text.startswith("=👩🏽‍💻")
    assert example.text[example.spans[0].start : example.spans[0].end] == "123"


def test_json_round_trip_preserves_intent_outputs_and_rejects_missing_intent():
    config = populated()
    switch = next(node for node in config.flow.nodes if node.type == "switch_intent")
    end = next(node for node in config.flow.nodes if node.type == "end")
    raw = config.model_dump(mode="json")
    raw["flow"]["edges"].append(
        {
            "id": str(uuid4()),
            "source": str(switch.id),
            "source_handle": str(config.intents[0].id),
            "target": str(end.id),
        }
    )
    parsed, errors = parse_json(encode(raw))
    assert not errors and parsed is not None
    restored, errors = parse_json(export_json(parsed))
    assert not errors and restored == parsed
    raw["intents"] = []
    assert parse_json(encode(raw))[0] is None


def test_nlu_settings_survive_export_and_merge_preserves_current_settings():
    current = populated()
    current.nlu_settings.score = 0.72
    current.quality_settings.intent_f1 = 0.9
    parsed, errors = parse_json(export_json(current))
    assert not errors and parsed is not None and parsed.nlu_settings.score == 0.72
    assert parsed.quality_settings.intent_f1 == 0.9
    legacy = current.model_dump(mode="json")
    del legacy["nlu_settings"]
    del legacy["quality_settings"]
    imported, errors = parse_json(encode(legacy))
    assert not errors and imported is not None and imported.nlu_settings.score == 0.55
    imported.intents.clear()
    imported.entities.clear()
    imported.examples.clear()
    preview = preview_json(imported, current, "merge")
    assert preview.config is not None and preview.config.nlu_settings.score == 0.72
    assert preview.config.quality_settings.intent_f1 == 0.9
    assert any(item.code == "settings_not_imported" for item in preview.warnings)
    invalid = current.model_dump(mode="json")
    for value in (-0.01, 1.01, True, "0.5", None):
        invalid["nlu_settings"]["score"] = value
        assert parse_json(encode(invalid))[0] is None


def test_json_rejects_unsupported_and_secret_fields_without_echoing_them():
    config = populated()
    raw = config.model_dump(mode="json")
    raw["flow"]["nodes"][0]["config"]["private_token"] = "secret-value"
    parsed, errors = parse_json(encode(raw))
    assert parsed is None and errors
    assert "private_token" not in repr(errors) and "secret-value" not in repr(errors)
    config.flow.nodes[0].config["private_token"] = "secret-value"
    with pytest.raises(ValueError):
        export_json(config)


def test_json_rejects_invalid_json_versions_and_unrecognized_fields():
    assert parse_json(b'{"schema_version":1,"schema_version":1}')[1][0].code == "invalid_json"
    assert parse_json(b'{"schema_version":NaN}')[1][0].code == "invalid_json"
    assert parse_json(b'{"text":"\\ud800"}')[1][0].code == "invalid_json"
    assert parse_json(b'{"position":1e999}')[1][0].code == "invalid_json"
    raw = populated().model_dump(mode="json")
    for version in (True, 2):
        assert (
            parse_json(json.dumps({"schema_version": version, "config": raw}).encode())[0] is None
        )
    raw["flow"]["schema_version"] = True
    assert parse_json(encode(raw))[0] is None
    raw["flow"] = None
    assert parse_json(encode(raw))[0] is None
    raw = populated().model_dump(mode="json")
    raw["PRIVATE_ORG_ID"] = "secret-value"
    _, errors = parse_json(encode(raw))
    assert errors and "PRIVATE_ORG_ID" not in repr(errors)


def test_json_rejects_merged_configuration_over_size_limit(monkeypatch):
    current = populated()
    monkeypatch.setattr("dialogentis.datasets.json_transfer.MAX_FILE_BYTES", 10)
    preview = preview_json(current, current, "replace")
    assert preview.config is None
    assert any(issue.code == "configuration_too_large" for issue in preview.errors)


def test_json_preview_blocks_conflicts_and_missing_references():
    current = populated()
    original = current.model_dump()
    duplicate = preview_json(current, current, "merge")
    assert duplicate.config is None and any(
        issue.code == "duplicate_id" for issue in duplicate.errors
    )
    assert any(issue.code == "duplicate_code" for issue in duplicate.errors)
    assert current.model_dump() == original
    unrelated = PortableConfig.model_validate(empty_config())
    warning = preview_json(unrelated, current, "merge")
    assert warning.config == current and warning.warnings[0].code == "flow_not_imported"
    broken = current.model_dump(mode="json")
    broken["examples"][0]["intent_id"] = str(uuid4())
    imported, errors = parse_json(encode(broken))
    assert imported is not None and not errors
    invalid = preview_json(imported, current, "replace")
    assert invalid.config is None and any(
        issue.code == "unknown_intent" for issue in invalid.errors
    )


def test_json_merge_adds_distinct_data_without_changing_either_input():
    current = populated()
    data = current.model_dump(mode="json")
    other = deepcopy(data)
    other["intents"][0]["id"] = str(uuid4())
    other["intents"][0]["code"] = "delivery"
    other["entities"][0]["id"] = str(uuid4())
    other["entities"][0]["code"] = "parcel"
    other["examples"][0]["id"] = str(uuid4())
    other["examples"][0]["intent_id"] = other["intents"][0]["id"]
    other["examples"][0]["spans"][0]["entity_id"] = other["entities"][0]["id"]
    other["examples"][0]["text"] = other["examples"][0]["text"].replace("123", "456")
    other["flow"] = empty_config()["flow"]
    imported = PortableConfig.model_validate(other)
    preview = preview_json(imported, current, "merge")
    assert not preview.errors and preview.config is not None
    assert preview.config.flow == current.flow and preview.warnings[0].code == "flow_not_imported"
    assert (
        len(preview.config.intents)
        == len(preview.config.entities)
        == len(preview.config.examples)
        == 2
    )
    assert current.model_dump(mode="json") == data and imported.model_dump(mode="json") == other
