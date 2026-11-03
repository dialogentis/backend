from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from dialogentis.nlu.tokenization import token_spans


@dataclass(frozen=True)
class Issue:
    path: tuple[str | int, ...]
    code: str
    message: str
    related_index: int | None = None


@dataclass(frozen=True)
class Annotation:
    entity_id: UUID
    start: int
    end: int
    text: str


def normalize_example(text: str) -> str:
    return " ".join(text.split()).casefold()


def duplicate_examples(examples: Sequence[tuple[str, Literal["train", "test"]]]) -> list[Issue]:
    seen: dict[str, tuple[int, str]] = {}
    issues: list[Issue] = []
    for index, (text, split) in enumerate(examples):
        normalized = normalize_example(text)
        if normalized in seen:
            previous, previous_split = seen[normalized]
            across_splits = split != previous_split
            issues.append(
                Issue(
                    ("examples", index, "text"),
                    "duplicate_across_splits" if across_splits else "duplicate_within_split",
                    "Фраза встречается в обучающем и тестовом наборах"
                    if across_splits
                    else "Фраза уже есть в наборе",
                    related_index=previous,
                )
            )
        else:
            seen[normalized] = (index, split)
    return issues


def validate_annotations(
    text: str,
    annotations: Sequence[Annotation],
    entity_methods: Mapping[UUID, Literal["learned", "dictionary", "regex"]],
) -> list[Issue]:
    issues: list[Issue] = []
    tokens = token_spans(text)
    starts = {start for start, _ in tokens}
    ends = {end for _, end in tokens}
    valid_ranges: list[tuple[int, int, int]] = []
    for index, annotation in enumerate(annotations):
        path = ("spans", index)
        method = entity_methods.get(annotation.entity_id)
        if method is None:
            issues.append(Issue((*path, "entity_id"), "unknown_entity", "Сущность не найдена"))
        if not 0 <= annotation.start < annotation.end <= len(text):
            issues.append(Issue(path, "invalid_range", "Границы разметки вне текста"))
            continue
        valid_ranges.append((annotation.start, annotation.end, index))
        if text[annotation.start : annotation.end] != annotation.text:
            issues.append(
                Issue((*path, "text"), "text_mismatch", "Текст разметки не совпадает с выделением")
            )
        if method == "learned" and (annotation.start not in starts or annotation.end not in ends):
            issues.append(
                Issue(path, "token_boundary", "Расширьте выделение до границ слов или символов")
            )
    previous_end = -1
    previous_index = -1
    for start, end, index in sorted(valid_ranges):
        if start < previous_end:
            issues.append(
                Issue(
                    ("spans", index), "overlap", "Сущности не должны пересекаться", previous_index
                )
            )
        if end > previous_end:
            previous_end = end
            previous_index = index
    return issues
