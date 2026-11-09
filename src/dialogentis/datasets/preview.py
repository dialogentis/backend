import asyncio
import math
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

import re2
from pydantic import BaseModel

from dialogentis.datasets.schema import Entity
from dialogentis.datasets.validation import normalize_example
from dialogentis.nlu.tokenization import token_spans


class Match(BaseModel):
    start: int
    end: int
    text: str
    value: str | int | float | bool | None
    error: Literal["invalid_value"] | None = None


class Preview(BaseModel):
    matches: list[Match]
    truncated: bool


async def run_preview(entity: Entity, text: str, slots: asyncio.Semaphore) -> Preview:
    async with asyncio.timeout(0.1):
        await slots.acquire()
    task = asyncio.create_task(asyncio.to_thread(preview_entity, entity, text))

    def release(completed: asyncio.Task[Preview]) -> None:
        slots.release()
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(release)
    async with asyncio.timeout(3):
        return await asyncio.shield(task)


def convert_value(value: str, value_type: str) -> str | int | float | bool:
    value = value.strip()
    if value_type == "string":
        return value
    if value_type == "integer":
        if not value or not value.lstrip("+-").isdecimal():
            raise ValueError("Invalid integer")
        number = int(value)
        if abs(number) > 2**53 - 1:
            raise ValueError("Integer exceeds exact JSON range")
        return number
    if value_type == "number":
        try:
            decimal = Decimal(value)
            number = float(decimal)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError("Invalid number") from exc
        if not math.isfinite(number) or (number == 0 and decimal != 0):
            raise ValueError("Invalid number")
        return number
    if value_type == "boolean":
        normalized = value.casefold()
        if normalized in {"true", "да"}:
            return True
        if normalized in {"false", "нет"}:
            return False
        raise ValueError("Invalid boolean")
    if value_type == "date":
        return date.fromisoformat(value).isoformat()
    raise ValueError("Invalid value type")


def preview_entity(
    entity: Entity, text: str, *, include_overlaps: bool = False, limit: int = 100
) -> Preview:
    if limit < 1:
        raise ValueError("Match limit must be positive")
    found: list[tuple[int, int, str]] = []
    if entity.method == "regex":
        if not entity.pattern:
            raise ValueError("Missing pattern")
        options = re2.Options()
        options.log_errors = False
        options.max_mem = 8 * 1024 * 1024
        try:
            compiled = re2.compile(entity.pattern, options=options)
            if isinstance(entity.group, int) and not 0 <= entity.group <= compiled.groups:
                raise ValueError("Invalid group")
            if isinstance(entity.group, str) and entity.group not in compiled.groupindex:
                raise ValueError("Invalid group")
            group = (
                compiled.groupindex[entity.group] if isinstance(entity.group, str) else entity.group
            )
            for match in compiled.finditer(text):
                start, end = match.span(group)
                if 0 <= start < end:
                    found.append((start, end, text[start:end]))
                    if len(found) > limit:
                        break
        except (re2.error, UnicodeError) as exc:
            raise ValueError("Invalid pattern") from exc
    elif entity.method == "dictionary":
        trie: dict = {}
        for entry in entity.dictionary:
            for term in (entry.value, *entry.synonyms):
                tokens = token_spans(term)
                if not tokens:
                    continue
                current = trie
                for start, end in tokens:
                    current = current.setdefault(term[start:end].casefold(), {})
                current.setdefault(None, []).append((normalize_example(term), entry.value))
        tokens = token_spans(text)
        index = 0
        while index < len(tokens):
            current = trie
            best: tuple[int, int, str] | None = None
            for end_index in range(index, len(tokens)):
                start, end = tokens[end_index]
                current = current.get(text[start:end].casefold())
                if current is None:
                    break
                original_start = tokens[index][0]
                normalized = normalize_example(text[original_start:end])
                for term, value in current.get(None, []):
                    if normalized == term:
                        candidate = (original_start, end, value)
                        if include_overlaps:
                            found.append(candidate)
                            if len(found) > limit:
                                break
                        if best is None or end > best[1]:
                            best = candidate
                if len(found) > limit:
                    break
            if include_overlaps:
                if len(found) > limit:
                    break
                index += 1
                continue
            if best is None:
                index += 1
            else:
                found.append(best)
                if len(found) > limit:
                    break
                while index < len(tokens) and tokens[index][0] < best[1]:
                    index += 1
    else:
        raise ValueError("Unsupported method")
    results: list[Match] = []
    for start, end, raw in found[:limit]:
        try:
            value = convert_value(raw, entity.value_type)
            results.append(Match(start=start, end=end, text=text[start:end], value=value))
        except ValueError:
            results.append(
                Match(start=start, end=end, text=text[start:end], value=None, error="invalid_value")
            )
    return Preview(matches=results, truncated=len(found) > limit)
