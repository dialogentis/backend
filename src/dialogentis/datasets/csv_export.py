import csv
import io
import json
import re
from unicodedata import normalize

from dialogentis.datasets.csv_import import MAX_FILE_BYTES
from dialogentis.datasets.schema import Dataset


def safe_cell(value: str) -> str:
    visible = re.sub(r"^[\s\ufeff]+", "", normalize("NFKC", value))
    if value.startswith(("\t", "\r", "\n")) or visible.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_csv(dataset: Dataset) -> bytes:
    intents = {intent.id: intent.code for intent in dataset.intents}
    entities = {entity.id: entity.code for entity in dataset.entities}
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["text", "intent", "split", "entities"])
    for example in dataset.examples:
        if example.intent_id is not None and example.intent_id not in intents:
            raise ValueError("Unknown intent")
        text = safe_cell(example.text)
        shift = len(text) - len(example.text)
        annotations = []
        for span in example.spans:
            if span.entity_id not in entities:
                raise ValueError("Unknown entity")
            annotations.append(
                {
                    "entity": entities[span.entity_id],
                    "start": span.start + shift,
                    "end": span.end + shift,
                }
            )
        writer.writerow(
            [
                text,
                intents[example.intent_id] if example.intent_id else "",
                example.split,
                json.dumps(annotations, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    try:
        content = output.getvalue().encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Invalid text") from exc
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Export exceeds file limit")
    return content
