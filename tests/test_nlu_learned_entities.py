from uuid import UUID

import pycrfsuite
import pytest

from dialogentis.datasets.schema import Entity, Span
from dialogentis.nlu.learned_entities import (
    bio_labels,
    decode_bio,
    features,
    train_learned_entities,
)
from dialogentis.nlu.tokenization import token_spans
from tests.test_nlu_intents import training_dataset


def learned_dataset():
    data = training_dataset()
    city = UUID("00000000-0000-0000-0000-000000000011")
    data.entities.append(
        Entity(id=city, code="city", name="Город", value_type="string", method="learned")
    )
    for index, example in enumerate(data.examples[:5]):
        example.text = f"👩🏽‍💻 доставьте в Москва номер {index}"
        start = example.text.index("Москва")
        example.spans = [Span(entity_id=city, start=start, end=start + len("Москва"))]
    return data, city


def test_bio_roundtrip_preserves_unicode_and_repairs_orphan_i():
    data, city = learned_dataset()
    example = data.examples[0]
    tokens = token_spans(example.text)
    labels = bio_labels(example, tokens, {city})
    assert decode_bio(tokens, labels, {city}) == [
        (city, example.spans[0].start, example.spans[0].end)
    ]
    text = "👩🏽‍💻 Новый   город, и ещё"
    parts = token_spans(text)
    labels = ["O", f"I:{city}", f"I:{city}", "O", f"I:{city}", "O"]
    spans = decode_bio(parts, labels, {city})
    assert [text[start:end] for _, start, end in spans] == ["Новый   город", "и"]
    assert len(features(text, parts)) == len(parts)
    with pytest.raises(ValueError):
        decode_bio(parts, labels[:-1], {city})
    with pytest.raises(ValueError):
        decode_bio(parts, [f"B:{UUID(int=100)}"] * len(parts), {city})


def test_crf_trains_only_on_train_and_extracts_with_original_offsets():
    data, city = learned_dataset()
    model = train_learned_entities(data)
    assert model is not None
    assert model.manifest["training_examples"] == 11
    tagger = pycrfsuite.Tagger()  # ty: ignore[unresolved-attribute]
    tagger.open_inmemory(model.model_bytes)
    assert not any("секретноетестовоеслово" in attribute for attribute in tagger.info().attributes)
    text = "👩🏽‍💻 доставьте в Москва номер 8"
    matches = model.predict(text)
    assert [(match.entity_id, match.text, match.value) for match in matches] == [
        (city, "Москва", "Москва")
    ]
    assert matches[0].start == len("👩🏽‍💻 доставьте в ")
    assert text[matches[0].start : matches[0].end] == matches[0].text
    assert model.predict(" ") == ()
    model.value_types[city] = "integer"
    assert model.predict(text)[0].error == "invalid_value"
    assert model.predict(text)[0].value is None


def test_no_learned_entities_needs_no_crf_and_hash_excludes_test():
    assert train_learned_entities(training_dataset()) is None
    data, _ = learned_dataset()
    first = train_learned_entities(data)
    assert first is not None
    data.examples.reverse()
    data.examples[0].text = "уникальная новая тестовая фраза"
    second = train_learned_entities(data)
    assert second is not None
    assert first.manifest["training_data_sha256"] == second.manifest["training_data_sha256"]


def test_learned_model_rejects_invalid_data_and_invalid_input():
    data, city = learned_dataset()
    data.examples[0].spans[0].start += 1
    with pytest.raises(ValueError, match="Dataset is not valid"):
        train_learned_entities(data)
    data, _ = learned_dataset()
    model = train_learned_entities(data)
    assert model is not None
    for text in ("x" * 2001, "\ud800"):
        with pytest.raises(ValueError):
            model.predict(text)
