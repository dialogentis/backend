import csv
import io
import json
from copy import deepcopy
from uuid import uuid4

from dialogentis.agents import empty_config
from dialogentis.datasets.import_worker import analyze_import


def csv_file(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["text", "intent", "split", "entities"])
    writer.writerows(rows)
    return output.getvalue().encode()


def test_csv_import_produces_complete_candidate_and_reports_errors():
    config = empty_config()
    config["intents"] = [{"id": str(uuid4()), "code": "order", "name": "Заказ"}]
    content = csv_file([["Заказ 42", "order", "train", "[]"]])
    candidate, report = analyze_import(content, "csv", "merge", config)
    assert candidate is not None and candidate["examples"][0]["text"] == "Заказ 42"
    assert candidate["flow"] == config["flow"] and config["examples"] == []
    assert report["counts"]["examples"] == 1 and not report["errors"]
    candidate, report = analyze_import(
        csv_file([["Текст", "missing", "test", "[]"]]), "csv", "merge", config
    )
    assert candidate is None and report["errors"][0]["code"] == "unknown_intent"


def test_csv_preview_rejects_existing_dangling_scenario_branch():
    config = empty_config()
    config["intents"] = [{"id": str(uuid4()), "code": "order", "name": "Заказ"}]
    switch, end = config["flow"]["nodes"][1], config["flow"]["nodes"][3]
    config["flow"]["edges"].append(
        {
            "id": str(uuid4()),
            "source": switch["id"],
            "source_handle": str(uuid4()),
            "target": end["id"],
        }
    )
    candidate, report = analyze_import(
        csv_file([["Заказ 42", "order", "train", "[]"]]), "csv", "merge", config
    )
    assert candidate is None
    assert any(item["code"] == "unknown_intent_output" for item in report["errors"])


def test_json_import_reports_code_conflicts_without_echoing_file_fields():
    config = empty_config()
    config["intents"] = [{"id": str(uuid4()), "code": "order", "name": "Заказ"}]
    imported = empty_config()
    imported["intents"] = [{"id": str(uuid4()), "code": "order", "name": "Другой"}]
    content = json.dumps({"schema_version": 1, "config": imported}, ensure_ascii=False).encode()
    candidate, report = analyze_import(content, "json", "merge", config)
    assert candidate is None and report["existing_codes"]["intents"] == ["order"]
    assert any(issue["code"] == "duplicate_code" for issue in report["errors"])
    assert report["warnings"][0]["code"] == "flow_not_imported"
    invalid = json.dumps({"schema_version": 1, "PRIVATE_SECRET": "hidden"}).encode()
    candidate, report = analyze_import(invalid, "json", "merge", config)
    assert (
        candidate is None and "PRIVATE_SECRET" not in repr(report) and "hidden" not in repr(report)
    )


def test_json_replace_cannot_remove_slot_still_referenced_by_imported_flow():
    config = empty_config()
    config["slots"] = [{"id": str(uuid4()), "code": "city"}]
    config["flow"]["nodes"][2]["config"]["text"] = "Город: {{city}}"
    imported = deepcopy(config)
    imported["slots"] = []
    content = json.dumps({"schema_version": 1, "config": imported}).encode()
    candidate, report = analyze_import(content, "json", "replace", config)
    assert candidate is None
    assert any(item["code"] == "slot_in_use" for item in report["errors"])


def test_json_import_marks_truncated_code_lists():
    config = empty_config()
    imported = empty_config()
    imported["entities"] = [
        {
            "id": str(uuid4()),
            "code": f"entity_{index}",
            "name": "Сущность",
            "value_type": "string",
            "method": "learned",
        }
        for index in range(1001)
    ]
    content = json.dumps({"schema_version": 1, "config": imported}).encode()
    candidate, report = analyze_import(content, "json", "merge", config)
    assert candidate is not None and report["counts"]["entities"] == 1001
    assert report["codes_truncated"] and len(report["new_codes"]["entities"]) == 1000
