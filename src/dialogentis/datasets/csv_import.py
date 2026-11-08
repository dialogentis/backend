import csv
import io
import json
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from pydantic import Field, StrictInt, ValidationError, model_validator

from dialogentis.contracts import InputModel
from dialogentis.datasets.schema import Dataset, Example, Span, validate_dataset

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 13000
MAX_ERRORS = 1000
HEADERS = frozenset({"text", "intent", "split", "entities"})
csv.field_size_limit(MAX_FILE_BYTES)


class CsvEntity(InputModel):
    entity: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def check_range(self):
        if self.start >= self.end:
            raise ValueError("Invalid range")
        return self


class CsvExample(InputModel):
    text: str = Field(min_length=1, max_length=2000)
    intent: str | None = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    split: Literal["train", "test"]
    entities: list[CsvEntity]

    @model_validator(mode="after")
    def check_intent(self):
        if self.intent is None and self.split != "test":
            raise ValueError("Fallback examples must be test examples")
        return self


@dataclass(frozen=True)
class CsvIssue:
    row: int
    field: str
    code: str


@dataclass(frozen=True)
class CsvRow:
    line: int
    example: CsvExample


@dataclass(frozen=True)
class CsvImport:
    rows: list[CsvRow]
    errors: list[CsvIssue]


@dataclass(frozen=True)
class CsvPreview:
    dataset: Dataset | None
    errors: list[CsvIssue]


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def parse_csv(content: bytes) -> CsvImport:
    if len(content) > MAX_FILE_BYTES:
        return CsvImport([], [CsvIssue(0, "file", "file_too_large")])
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return CsvImport([], [CsvIssue(0, "file", "invalid_utf8")])

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    rows: list[CsvRow] = []
    errors: list[CsvIssue] = []
    try:
        header = next(reader, [])
        if len(header) != len(HEADERS) or set(header) != HEADERS:
            return CsvImport([], [CsvIssue(1, "header", "invalid_header")])
        count = 0
        while True:
            row_number = reader.line_num + 1
            if len(errors) >= MAX_ERRORS:
                errors.append(CsvIssue(row_number, "file", "too_many_errors"))
                break
            values = next(reader, None)
            if values is None:
                break
            if not values:
                continue
            count += 1
            if count > MAX_ROWS:
                errors.append(CsvIssue(row_number, "file", "row_limit"))
                break
            if len(values) != len(header):
                errors.append(CsvIssue(row_number, "row", "invalid_columns"))
                continue
            record = dict(zip(header, values, strict=True))
            try:
                entities = json.loads(record["entities"], object_pairs_hook=unique_object)
            except ValueError, RecursionError:
                errors.append(CsvIssue(row_number, "entities", "invalid_json"))
                continue
            if (
                not isinstance(entities, list)
                or len(entities) > 2000
                or any(
                    not isinstance(value, dict) or set(value) != {"entity", "start", "end"}
                    for value in entities
                )
            ):
                errors.append(CsvIssue(row_number, "entities", "invalid_entities"))
                continue
            try:
                rows.append(
                    CsvRow(
                        line=row_number,
                        example=CsvExample.model_validate(
                            {**record, "intent": record["intent"] or None, "entities": entities}
                        ),
                    )
                )
            except ValidationError as exc:
                for error in exc.errors(
                    include_input=False, include_context=False, include_url=False
                ):
                    path = error["loc"]
                    field = str(path[0]) if path else "row"
                    errors.append(CsvIssue(row_number, field, error["type"]))
                    if len(errors) >= MAX_ERRORS:
                        break
    except csv.Error:
        errors.append(CsvIssue(reader.line_num or 1, "file", "invalid_csv"))
    return CsvImport(rows, errors)


def preview_csv(
    parsed: CsvImport, current: Dataset, mode: Literal["merge", "replace"]
) -> CsvPreview:
    if mode not in {"merge", "replace"}:
        raise ValueError("Invalid import mode")
    if parsed.errors:
        return CsvPreview(None, parsed.errors)
    if not parsed.rows:
        return CsvPreview(None, [CsvIssue(0, "file", "empty_file")])
    intents = {intent.code: intent.id for intent in current.intents}
    entities = {entity.code: entity.id for entity in current.entities}
    imported: list[Example] = []
    errors: list[CsvIssue] = []
    for row in parsed.rows:
        item = row.example
        row_errors = len(errors)
        intent_id = intents.get(item.intent) if item.intent else None
        if item.intent and intent_id is None:
            errors.append(CsvIssue(row.line, "intent", "unknown_intent"))
        spans: list[Span] = []
        for value in item.entities:
            entity_id = entities.get(value.entity)
            if entity_id is None:
                errors.append(CsvIssue(row.line, "entities", "unknown_entity"))
            else:
                spans.append(Span(entity_id=entity_id, start=value.start, end=value.end))
        if len(errors) == row_errors:
            imported.append(
                Example(
                    id=uuid4(),
                    text=item.text,
                    intent_id=intent_id,
                    expected_fallback=item.intent is None,
                    split=item.split,
                    spans=spans,
                )
            )
        if len(errors) >= MAX_ERRORS:
            return CsvPreview(
                None, [*errors[:MAX_ERRORS], CsvIssue(row.line, "file", "too_many_errors")]
            )
    if errors:
        return CsvPreview(None, errors)
    existing = current.examples if mode == "merge" else []
    try:
        combined = Dataset(
            intents=current.intents, entities=current.entities, examples=[*existing, *imported]
        )
    except ValidationError:
        return CsvPreview(None, [CsvIssue(0, "file", "example_limit")])
    issues, _ = validate_dataset(combined)
    readiness = {
        "insufficient_intents",
        "insufficient_intent_examples",
        "insufficient_entity_examples",
    }
    for issue in issues:
        if issue.code in readiness:
            continue
        line, field = 0, "file"
        if len(issue.path) > 1 and issue.path[0] == "examples" and isinstance(issue.path[1], int):
            index = issue.path[1] - len(existing)
            if 0 <= index < len(parsed.rows):
                line = parsed.rows[index].line
                field = str(issue.path[2]) if len(issue.path) > 2 else "row"
                field = {"intent_id": "intent", "spans": "entities"}.get(field, field)
        errors.append(CsvIssue(line, field, issue.code))
        if len(errors) >= MAX_ERRORS:
            errors.append(CsvIssue(line, "file", "too_many_errors"))
            break
    return CsvPreview(None if errors else combined, errors)
