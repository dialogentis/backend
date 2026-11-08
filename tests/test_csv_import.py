import csv
import io
import json
from uuid import uuid4

from dialogentis.datasets.csv_import import MAX_FILE_BYTES, parse_csv, preview_csv
from dialogentis.datasets.schema import Dataset, Entity, Example, Intent


def csv_file(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["text", "intent", "split", "entities"])
    writer.writerows(rows)
    return output.getvalue().encode()


def test_csv_quotes_newlines_and_unicode_offsets():
    text = '📦 "заказ"\n123'
    entities = json.dumps([{"entity": "order", "start": 10, "end": 13}])
    result = parse_csv(csv_file([[text, "order_status", "train", entities]]))
    assert not result.errors
    example = result.rows[0].example
    assert example.text == text
    assert text[example.entities[0].start : example.entities[0].end] == "123"


def test_csv_errors_are_addressed_and_do_not_expose_values():
    result = parse_csv(
        csv_file(
            [
                ["valid", "order_status", "train", "[]"],
                ["secret", "", "train", "[]"],
                ["secret", "order_status", "test", '[{"entity":"order","start":true,"end":2}]'],
                ["secret", "order_status", "test", '[{"entity":"x","entity":"y"}]'],
            ]
        )
    )
    assert len(result.rows) == 1
    assert [(error.row, error.field) for error in result.errors] == [
        (3, "row"),
        (4, "entities"),
        (5, "entities"),
    ]
    assert "secret" not in repr(result.errors)


def test_rejects_invalid_files():
    assert parse_csv(b"\xff").errors[0].code == "invalid_utf8"
    assert parse_csv(b"x" * (MAX_FILE_BYTES + 1)).errors[0].code == "file_too_large"
    assert parse_csv(b"text,intent,split,entities,secret\n").errors[0].code == "invalid_header"
    assert parse_csv(b'text,intent,split,entities\n"unfinished').errors[0].code == "invalid_csv"


def test_preview_validates_codes_annotations_and_duplicates_atomically():
    intent = Intent(id=uuid4(), code="order_status", name="Заказ")
    entity = Entity(id=uuid4(), code="order", name="Номер", value_type="string", method="learned")
    existing = Example(id=uuid4(), text="Заказ 123", intent_id=intent.id, split="train")
    current = Dataset(intents=[intent], entities=[entity], examples=[existing])
    valid = parse_csv(
        csv_file(
            [["Посылка 42", "order_status", "test", '[{"entity":"order","start":8,"end":10}]']]
        )
    )
    preview = preview_csv(valid, current, "merge")
    assert not preview.errors and preview.dataset is not None
    assert len(preview.dataset.examples) == 2 and len(current.examples) == 1
    assert preview.dataset.examples[1].spans[0].entity_id == entity.id
    duplicate = parse_csv(csv_file([["  заказ   123 ", "order_status", "test", "[]"]]))
    invalid = preview_csv(duplicate, current, "merge")
    assert invalid.dataset is None and invalid.errors[0].code == "duplicate_across_splits"
    assert invalid.errors[0].row == 2
    assert preview_csv(duplicate, current, "replace").dataset is not None
    unknown = parse_csv(
        csv_file([["фраза", "missing", "test", '[{"entity":"missing","start":0,"end":5}]']])
    )
    invalid = preview_csv(unknown, current, "merge")
    assert invalid.dataset is None
    assert {error.code for error in invalid.errors} == {"unknown_intent", "unknown_entity"}
    bad_span = parse_csv(
        csv_file([["заказ 42", "order_status", "test", '[{"entity":"order","start":7,"end":9}]']])
    )
    assert preview_csv(bad_span, current, "merge").dataset is None
    assert preview_csv(parse_csv(csv_file([])), current, "replace").dataset is None
