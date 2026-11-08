from uuid import uuid4

import pytest

from dialogentis.datasets.csv_export import export_csv, safe_cell
from dialogentis.datasets.csv_import import parse_csv, preview_csv
from dialogentis.datasets.schema import Dataset, Entity, Example, Intent, Span


def test_csv_export_guards_formulas_and_preserves_annotation_after_import():
    intent = Intent(id=uuid4(), code="order_status", name="Заказ")
    entity = Entity(id=uuid4(), code="order", name="Номер", value_type="string", method="learned")
    example = Example(
        id=uuid4(),
        text="=123",
        intent_id=intent.id,
        split="train",
        spans=[Span(entity_id=entity.id, start=1, end=4)],
    )
    current = Dataset(intents=[intent], entities=[entity], examples=[example])
    result = parse_csv(export_csv(current))
    assert not result.errors
    assert result.rows[0].example.text == "'=123"
    assert (result.rows[0].example.entities[0].start, result.rows[0].example.entities[0].end) == (
        2,
        5,
    )
    preview = preview_csv(result, current, "replace")
    assert preview.dataset is not None and not preview.errors
    assert current.examples[0].text == "=123"
    for text in ["=SUM(A1)", "+1", "-1", "@cmd", "  =cmd", "\ttext", "\rcmd"]:
        assert safe_cell(text) == "'" + text
    assert safe_cell("Заказ,\n123") == "Заказ,\n123"


def test_csv_export_rejects_unknown_references():
    example = Example(id=uuid4(), text="Заказ", intent_id=uuid4(), split="train")
    with pytest.raises(ValueError, match="Unknown intent"):
        export_csv(Dataset(intents=[], entities=[], examples=[example]))


def test_csv_export_checks_the_size_before_returning_data(monkeypatch):
    monkeypatch.setattr("dialogentis.datasets.csv_export.MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="file limit"):
        export_csv(Dataset(intents=[], entities=[], examples=[]))
