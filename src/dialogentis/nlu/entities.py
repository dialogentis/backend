"""Combine entity extractors without losing overlapping candidates."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from dialogentis.datasets.preview import preview_entity
from dialogentis.datasets.schema import Entity
from dialogentis.nlu.learned_entities import LearnedEntityModel

Source = Literal["regex", "dictionary", "learned"]
MAX_CANDIDATES = 10000


@dataclass(frozen=True)
class EntityMatch:
    entity_id: UUID
    source: Source
    start: int
    end: int
    text: str
    value: str | int | float | bool | None
    error: str | None = None


@dataclass(frozen=True)
class RejectedMatch:
    candidate: EntityMatch
    overlaps: EntityMatch


@dataclass(frozen=True)
class EntityPrediction:
    matches: tuple[EntityMatch, ...]
    rejected: tuple[RejectedMatch, ...]


def extract_entities(
    entities: list[Entity], learned_model: LearnedEntityModel | None, text: str
) -> EntityPrediction:
    if len(text) > 2000:
        raise ValueError("Text exceeds 2000 code points")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Text is not valid Unicode") from exc
    if len({item.id for item in entities}) != len(entities):
        raise ValueError("Duplicate entity IDs")
    learned_types = {item.id: item.value_type for item in entities if item.method == "learned"}
    if (learned_model.value_types if learned_model else {}) != learned_types:
        raise ValueError("Learned model does not match entity definitions")
    candidates: list[EntityMatch] = []
    for entity in entities:
        if entity.method == "learned":
            continue
        preview = preview_entity(entity, text, include_overlaps=True, limit=MAX_CANDIDATES)
        if preview.truncated:
            raise ValueError("Entity candidate limit exceeded")
        candidates.extend(
            EntityMatch(
                entity.id,
                entity.method,
                match.start,
                match.end,
                match.text,
                match.value,
                match.error,
            )
            for match in preview.matches
        )
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError("Entity candidate limit exceeded")
    if learned_model:
        candidates.extend(
            EntityMatch(
                match.entity_id,
                "learned",
                match.start,
                match.end,
                match.text,
                match.value,
                match.error,
            )
            for match in learned_model.predict(text)
        )
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("Entity candidate limit exceeded")
    priority = {"regex": 0, "dictionary": 1, "learned": 2}
    candidates.sort(
        key=lambda item: (
            priority[item.source],
            -(item.end - item.start),
            str(item.entity_id),
            item.start,
            item.end,
        )
    )
    occupied = 0
    accepted: list[EntityMatch] = []
    rejected = []
    for candidate in candidates:
        mask = ((1 << (candidate.end - candidate.start)) - 1) << candidate.start
        if occupied & mask:
            blocker = next(
                item
                for item in accepted
                if item.start < candidate.end and candidate.start < item.end
            )
            rejected.append(RejectedMatch(candidate, blocker))
        else:
            accepted.append(candidate)
            occupied |= mask
    accepted.sort(key=lambda item: (item.start, item.end, str(item.entity_id)))
    return EntityPrediction(tuple(accepted), tuple(rejected))
