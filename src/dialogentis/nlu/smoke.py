"""Executable compatibility checks, not a published agent or quality benchmark."""

import tempfile
from pathlib import Path

import pycrfsuite
import re2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def check_nlu() -> dict[str, str]:
    classifier = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=0))
    classifier.fit(
        ["привет", "здравствуйте", "доставка", "адрес доставки"],
        ["greeting", "greeting", "delivery", "delivery"],
    )
    assert classifier.predict(["привет"])[0] == "greeting"
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "entity.crfsuite")
        # Cython exports these at runtime; upstream does not ship type information.
        trainer = pycrfsuite.Trainer(verbose=False)  # ty: ignore[unresolved-attribute]
        for _ in range(5):
            trainer.append(
                [{"word": "заказ"}, {"word": "12345", "digit": True}], ["O", "B-order_id"]
            )
        trainer.train(path)
        tagger = pycrfsuite.Tagger()  # ty: ignore[unresolved-attribute]
        tagger.open(path)
        assert tagger.tag([{"word": "заказ"}, {"word": "12345", "digit": True}]) == [
            "O",
            "B-order_id",
        ]
    assert re2.fullmatch(r"\d{3,10}", "12345") is not None
    options = re2.Options()
    options.log_errors = False
    for expression in [r"(a)\1", r"(?<=a)b"]:
        try:
            re2.compile(expression, options=options)
        except re2.error:
            continue
        raise AssertionError("Unsafe expression was accepted")
    return {"intent": "ok", "crf": "ok", "regex": "ok"}


if __name__ == "__main__":
    print(check_nlu())
