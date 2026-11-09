"""CRF BIO entity extraction over offsets in the original text."""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pycrfsuite

from dialogentis.datasets.preview import convert_value
from dialogentis.datasets.schema import Dataset, Example, validate_dataset
from dialogentis.nlu.intent import InvalidTrainingDataset
from dialogentis.nlu.tokenization import token_spans


@dataclass(frozen=True)
class LearnedMatch:
    entity_id: UUID
    start: int
    end: int
    text: str
    value: str | int | float | bool | None
    error: str | None = None


def features(text: str, tokens: list[tuple[int, int]]) -> list[dict]:
    words = [text[start:end] for start, end in tokens]
    output = []
    for index, word in enumerate(words):
        item = {
            "bias": 1.0,
            "word.lower": word.casefold(),
            "word.prefix2": word[:2].casefold(),
            "word.suffix2": word[-2:].casefold(),
            "word.suffix3": word[-3:].casefold(),
            "word.isdigit": word.isdigit(),
            "word.isupper": word.isupper(),
            "word.istitle": word.istitle(),
        }
        if index:
            item["prev.lower"] = words[index - 1].casefold()
            item["prev.istitle"] = words[index - 1].istitle()
        else:
            item["BOS"] = True
        if index + 1 < len(words):
            item["next.lower"] = words[index + 1].casefold()
            item["next.istitle"] = words[index + 1].istitle()
        else:
            item["EOS"] = True
        output.append(item)
    return output


def bio_labels(example: Example, tokens: list[tuple[int, int]], learned: set[UUID]) -> list[str]:
    annotations = sorted(
        (span for span in example.spans if span.entity_id in learned), key=lambda span: span.start
    )
    labels = []
    index = 0
    for start, end in tokens:
        while index < len(annotations) and annotations[index].end <= start:
            index += 1
        if (
            index < len(annotations)
            and annotations[index].start <= start
            and end <= annotations[index].end
        ):
            span = annotations[index]
            prefix = "B" if start == span.start else "I"
            labels.append(f"{prefix}:{span.entity_id}")
        else:
            labels.append("O")
    return labels


def decode_bio(
    tokens: list[tuple[int, int]], labels: list[str], known: set[UUID]
) -> list[tuple[UUID, int, int]]:
    if len(tokens) != len(labels):
        raise ValueError("Invalid CRF output length")
    result: list[tuple[UUID, int, int]] = []
    previous_entity: UUID | None = None
    for (start, end), label in zip(tokens, labels, strict=True):
        if label == "O":
            previous_entity = None
            continue
        prefix, separator, identifier = label.partition(":")
        if not separator or prefix not in {"B", "I"}:
            raise ValueError("Invalid CRF label")
        try:
            entity_id = UUID(identifier)
        except ValueError as exc:
            raise ValueError("Invalid CRF label") from exc
        if entity_id not in known:
            raise ValueError("Unknown CRF entity")
        if prefix == "I" and result and result[-1][0] == entity_id and previous_entity == entity_id:
            result[-1] = (entity_id, result[-1][1], end)
        else:
            result.append((entity_id, start, end))
        previous_entity = entity_id
    return result


@dataclass
class LearnedEntityModel:
    model_bytes: bytes
    value_types: dict[UUID, str]
    manifest: dict

    def predict(self, text: str) -> tuple[LearnedMatch, ...]:
        if len(text) > 2000:
            raise ValueError("Text exceeds 2000 code points")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Text is not valid Unicode") from exc
        tokens = token_spans(text)
        if not tokens:
            return ()
        # A Tagger holds mutable state, so each prediction uses its own instance.
        tagger = pycrfsuite.Tagger()  # ty: ignore[unresolved-attribute]
        tagger.open_inmemory(self.model_bytes)
        labels = tagger.tag(features(text, tokens))
        spans = decode_bio(tokens, labels, set(self.value_types))
        matches = []
        for entity_id, start, end in spans:
            raw = text[start:end]
            try:
                value = convert_value(raw, self.value_types[entity_id])
                matches.append(LearnedMatch(entity_id, start, end, raw, value))
            except ValueError:
                matches.append(LearnedMatch(entity_id, start, end, raw, None, "invalid_value"))
        return tuple(matches)


def training_examples(dataset: Dataset) -> list[Example]:
    return sorted(
        (item for item in dataset.examples if item.split == "train"),
        key=lambda item: (item.text, str(item.id)),
    )


def training_data_hash(dataset: Dataset) -> str:
    learned = {item.id for item in dataset.entities if item.method == "learned"}
    content = [
        (
            item.text,
            sorted(
                (str(span.entity_id), span.start, span.end)
                for span in item.spans
                if span.entity_id in learned
            ),
        )
        for item in training_examples(dataset)
    ]
    return hashlib.sha256(
        json.dumps(
            [sorted(map(str, learned)), content], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def train_learned_entities(dataset: Dataset) -> LearnedEntityModel | None:
    errors, _ = validate_dataset(dataset)
    if errors:
        raise InvalidTrainingDataset(errors)
    value_types: dict[UUID, str] = {
        item.id: item.value_type for item in dataset.entities if item.method == "learned"
    }
    if not value_types:
        return None
    learned = set(value_types)
    examples = training_examples(dataset)
    fingerprint = training_data_hash(dataset)
    # Cython exports Trainer at runtime but does not ship type information.
    trainer = pycrfsuite.Trainer(verbose=False)  # ty: ignore[unresolved-attribute]
    for example in examples:
        tokens = token_spans(example.text)
        trainer.append(features(example.text, tokens), bio_labels(example, tokens, learned))
    trainer.set_params(
        {"c1": 0.1, "c2": 0.1, "max_iterations": 100, "feature.possible_transitions": True}
    )
    with TemporaryDirectory(prefix="dialogentis-crf-") as directory:
        model_path = Path(directory) / "model.crfsuite"
        trainer.train(str(model_path))
        model_bytes = model_path.read_bytes()
    return LearnedEntityModel(
        model_bytes=model_bytes,
        value_types=value_types,
        manifest={
            "kind": "crfsuite-bio",
            "python_crfsuite_version": version("python-crfsuite"),
            "feature_set_version": 1,
            "training_examples": len(examples),
            "training_data_sha256": fingerprint,
            "algorithm": "lbfgs",
            "c1": 0.1,
            "c2": 0.1,
            "max_iterations": 100,
            "possible_transitions": True,
        },
    )
