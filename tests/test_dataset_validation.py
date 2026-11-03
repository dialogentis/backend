from typing import Literal
from uuid import uuid4

from dialogentis.datasets.validation import Annotation, duplicate_examples, validate_annotations
from dialogentis.nlu.tokenization import token_spans


def test_unicode_offsets_preserve_emoji_combining_marks_and_zwj():
    text = "Заказ 👩🏽‍💻 и е\u0308лка 🇷🇺!"
    tokens = [text[start:end] for start, end in token_spans(text)]
    assert tokens == ["Заказ", "👩🏽‍💻", "и", "е\u0308лка", "🇷🇺", "!"]
    entity = uuid4()
    start = text.index("👩")
    end = start + len("👩🏽‍💻")
    assert (
        validate_annotations(text, [Annotation(entity, start, end, "👩🏽‍💻")], {entity: "learned"})
        == []
    )
    errors = validate_annotations(
        text, [Annotation(entity, start, start + 1, "👩")], {entity: "learned"}
    )
    assert [issue.code for issue in errors] == ["token_boundary"]


def test_annotations_reject_overlap_unknown_entity_and_wrong_offsets():
    entity, unknown = uuid4(), uuid4()
    text = "Москва и Казань"
    annotations = [
        Annotation(entity, 0, 6, "Москва"),
        Annotation(unknown, 4, 9, "ошибка"),
        Annotation(entity, 99, 101, "X"),
    ]
    codes = {issue.code for issue in validate_annotations(text, annotations, {entity: "learned"})}
    assert codes == {"unknown_entity", "text_mismatch", "invalid_range", "overlap"}


def test_normalized_duplicates_are_reported_with_split_and_source():
    examples: list[tuple[str, Literal["train", "test"]]] = [
        ("  Привет\tМИР ", "train"),
        ("привет мир", "train"),
        ("ПРИВЕТ  мир", "test"),
        ("ёж", "test"),
        ("е\u0308ж", "test"),
    ]
    issues = duplicate_examples(examples)
    assert [(issue.path, issue.code, issue.related_index) for issue in issues] == [
        (("examples", 1, "text"), "duplicate_within_split", 0),
        (("examples", 2, "text"), "duplicate_across_splits", 0),
    ]
