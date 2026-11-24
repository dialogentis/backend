from collections import Counter
from typing import Literal
from uuid import UUID

import re2
from pydantic import Field, StrictBool, StrictInt

from dialogentis.contracts import InputModel
from dialogentis.datasets.validation import (
    Annotation,
    Issue,
    duplicate_examples,
    normalize_example,
    validate_annotations,
)


class Intent(InputModel):
    id: UUID
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    enabled: StrictBool = True


class DictionaryValue(InputModel):
    value: str = Field(min_length=1, max_length=2000)
    synonyms: list[str] = Field(default_factory=list)


class Entity(InputModel):
    id: UUID
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=120)
    value_type: Literal["string", "integer", "number", "boolean", "date"]
    method: Literal["learned", "dictionary", "regex"]
    dictionary: list[DictionaryValue] = Field(default_factory=list)
    pattern: str | None = Field(default=None, max_length=2048)
    group: StrictInt | str = 0


class Span(InputModel):
    entity_id: UUID
    start: StrictInt
    end: StrictInt
    text: str | None = None


class Example(InputModel):
    id: UUID
    text: str = Field(min_length=1, max_length=2000)
    intent_id: UUID | None = None
    expected_fallback: StrictBool = False
    split: Literal["train", "test"]
    spans: list[Span] = Field(default_factory=list)


class Dataset(InputModel):
    intents: list[Intent] = Field(max_length=100)
    entities: list[Entity]
    examples: list[Example] = Field(max_length=13000)


def validate_dataset(dataset: Dataset) -> tuple[list[Issue], list[Issue]]:
    errors: list[Issue] = []
    warnings: list[Issue] = []
    for collection in ("intents", "entities", "examples"):
        seen_ids: dict[UUID, int] = {}
        seen_codes: dict[str, int] = {}
        for index, item in enumerate(getattr(dataset, collection)):
            if item.id in seen_ids:
                errors.append(
                    Issue(
                        (collection, index, "id"),
                        "duplicate_id",
                        "Идентификатор повторяется",
                        seen_ids[item.id],
                    )
                )
            else:
                seen_ids[item.id] = index
            if isinstance(item, Intent | Entity):
                if item.code in seen_codes:
                    errors.append(
                        Issue(
                            (collection, index, "code"),
                            "duplicate_code",
                            "Код уже используется",
                            seen_codes[item.code],
                        )
                    )
                else:
                    seen_codes[item.code] = index

    intent_ids = {item.id for item in dataset.intents}
    methods = {item.id: item.method for item in dataset.entities}
    errors.extend(duplicate_examples([(item.text, item.split) for item in dataset.examples]))
    counts = Counter(item.split for item in dataset.examples)
    if counts["train"] > 10000:
        errors.append(Issue(("examples",), "train_limit", "Не более 10000 фраз для модели"))
    if counts["test"] > 3000:
        errors.append(Issue(("examples",), "test_limit", "Не более 3000 тестовых фраз"))
    for index, item in enumerate(dataset.examples):
        path = ("examples", index)
        if not item.text.strip():
            errors.append(Issue((*path, "text"), "empty_text", "Укажите текст фразы"))
        if item.expected_fallback:
            if item.split != "test" or item.intent_id is not None:
                errors.append(
                    Issue(
                        path,
                        "invalid_fallback",
                        "Неизвестная фраза допускается только в тесте и без намерения",
                    )
                )
        elif item.intent_id not in intent_ids:
            errors.append(Issue((*path, "intent_id"), "unknown_intent", "Намерение не найдено"))
        annotations = [
            Annotation(
                span.entity_id,
                span.start,
                span.end,
                span.text if span.text is not None else item.text[span.start : span.end],
            )
            for span in item.spans
        ]
        for issue in validate_annotations(item.text, annotations, methods):
            errors.append(
                Issue((*path, *issue.path), issue.code, issue.message, issue.related_index)
            )

    options = re2.Options()
    options.log_errors = False
    options.max_mem = 8 * 1024 * 1024
    for index, entity in enumerate(dataset.entities):
        path = ("entities", index)
        if not entity.name.strip():
            errors.append(Issue((*path, "name"), "empty_name", "Укажите название сущности"))
        if entity.method == "regex":
            if not entity.pattern:
                errors.append(Issue((*path, "pattern"), "missing_pattern", "Укажите шаблон"))
                continue
            try:
                compiled = re2.compile(entity.pattern, options=options)
                if compiled.search("") is not None:
                    errors.append(
                        Issue(
                            (*path, "pattern"),
                            "empty_regex_match",
                            "Шаблон не должен совпадать с пустой строкой",
                        )
                    )
                if isinstance(entity.group, int) and not 0 <= entity.group <= compiled.groups:
                    errors.append(
                        Issue((*path, "group"), "unknown_group", "Группа не найдена в шаблоне")
                    )
                elif isinstance(entity.group, str) and entity.group not in compiled.groupindex:
                    errors.append(
                        Issue((*path, "group"), "unknown_group", "Группа не найдена в шаблоне")
                    )
            except re2.error, UnicodeError:
                errors.append(
                    Issue((*path, "pattern"), "invalid_regex", "Шаблон не поддерживается RE2")
                )
        if entity.method == "dictionary" and not entity.dictionary:
            warnings.append(
                Issue((*path, "dictionary"), "empty_dictionary", "Добавьте значения словаря")
            )
        if entity.method == "dictionary":
            terms: dict[str, int] = {}
            for value_index, entry in enumerate(entity.dictionary):
                for term in [entry.value, *entry.synonyms]:
                    normalized = normalize_example(term)
                    if not normalized:
                        errors.append(
                            Issue(
                                (*path, "dictionary", value_index),
                                "empty_term",
                                "Значение словаря не должно быть пустым",
                            )
                        )
                    elif normalized in terms:
                        errors.append(
                            Issue(
                                (*path, "dictionary", value_index),
                                "duplicate_term",
                                "Значение или синоним повторяется",
                                terms[normalized],
                            )
                        )
                    else:
                        terms[normalized] = value_index
    for index, intent in enumerate(dataset.intents):
        if not intent.name.strip():
            errors.append(
                Issue(("intents", index, "name"), "empty_name", "Укажите название намерения")
            )
    return errors, warnings
