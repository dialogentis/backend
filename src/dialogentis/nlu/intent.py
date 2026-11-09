"""Intent classification, independent of HTTP, storage and the dialogue runtime."""

import hashlib
import json
from dataclasses import dataclass
from platform import python_version
from typing import Literal
from unicodedata import unidata_version
from uuid import UUID

import numpy
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from dialogentis.datasets.schema import Dataset, validate_dataset
from dialogentis.datasets.validation import Issue

FallbackReason = Literal["empty_features", "low_score", "ambiguous_intent", "low_similarity"]


@dataclass(frozen=True)
class Thresholds:
    score: float = 0.55
    margin: float = 0.15
    similarity: float = 0.20

    def __post_init__(self) -> None:
        if not all(0 <= value <= 1 for value in (self.score, self.margin, self.similarity)):
            raise ValueError("NLU thresholds must be between 0 and 1")


@dataclass(frozen=True)
class RankedIntent:
    intent_id: UUID
    score: float


@dataclass(frozen=True)
class IntentPrediction:
    ranking: tuple[RankedIntent, ...]
    similarity: float
    fallback_reasons: tuple[FallbackReason, ...]

    @property
    def intent_id(self) -> UUID | None:
        return None if self.fallback_reasons else self.ranking[0].intent_id

    @property
    def fallback_reason(self) -> FallbackReason | None:
        return self.fallback_reasons[0] if self.fallback_reasons else None


class InvalidTrainingDataset(ValueError):
    def __init__(self, issues: list[Issue]):
        super().__init__("Dataset is not valid for training")
        self.issues = tuple(issues)


@dataclass
class IntentModel:
    vectorizer: FeatureUnion
    classifier: LogisticRegression
    train_vectors: csr_matrix
    manifest: dict

    def predict(self, text: str, thresholds: Thresholds | None = None) -> IntentPrediction:
        thresholds = thresholds or Thresholds()
        if len(text) > 2000:
            raise ValueError("Text exceeds 2000 code points")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Text is not valid Unicode") from exc
        vector = normalize(self.vectorizer.transform([text]))
        scores = self.classifier.predict_proba(vector)[0]
        ranking = tuple(
            RankedIntent(UUID(str(label)), float(score))
            for label, score in sorted(
                zip(self.classifier.classes_, scores, strict=True),
                key=lambda item: (-item[1], str(item[0])),
            )
        )
        similarity = float((vector @ self.train_vectors.T).max())
        reasons: list[FallbackReason] = []
        if vector.nnz == 0:
            reasons.append("empty_features")
        if ranking[0].score < thresholds.score:
            reasons.append("low_score")
        if ranking[0].score - ranking[1].score < thresholds.margin:
            reasons.append("ambiguous_intent")
        if similarity < thresholds.similarity:
            reasons.append("low_similarity")
        return IntentPrediction(ranking, similarity, tuple(reasons))


def training_examples(dataset: Dataset):
    enabled = {intent.id for intent in dataset.intents if intent.enabled}
    return sorted(
        (item for item in dataset.examples if item.split == "train" and item.intent_id in enabled),
        key=lambda item: (str(item.intent_id), item.text, str(item.id)),
    )


def training_data_hash(dataset: Dataset) -> str:
    content = [(str(item.intent_id), item.text) for item in training_examples(dataset)]
    encoded = json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_intents(dataset: Dataset) -> IntentModel:
    errors, _ = validate_dataset(dataset)
    if errors:
        raise InvalidTrainingDataset(errors)
    train = training_examples(dataset)
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=25000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"
                ),
            ),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000)),
        ]
    )
    try:
        train_vectors = normalize(features.fit_transform([item.text for item in train]))
    except ValueError as exc:
        raise ValueError(
            "Training texts cannot produce the required word and character features"
        ) from exc
    classifier = LogisticRegression(
        random_state=42, max_iter=1000, solver="lbfgs", C=1.0, tol=0.0001
    )
    classifier.fit(train_vectors, [str(item.intent_id) for item in train])
    return IntentModel(
        vectorizer=features,
        classifier=classifier,
        train_vectors=train_vectors,
        manifest={
            "kind": "tfidf-logistic-regression",
            "python_version": python_version(),
            "unicode_version": unidata_version,
            "sklearn_version": sklearn.__version__,
            "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
            "seed": 42,
            "training_examples": len(train),
            "training_data_sha256": training_data_hash(dataset),
            "word_ngrams": [1, 2],
            "word_token_pattern": r"(?u)\b\w+\b|[^\w\s]",
            "char_wb_ngrams": [3, 5],
            "max_features_per_vectorizer": 25000,
            "lowercase": True,
            "tfidf_norm": "l2",
            "tfidf_smooth_idf": True,
            "tfidf_sublinear_tf": False,
            "vector_normalization": "l2",
            "classifier_solver": "lbfgs",
            "classifier_c": 1.0,
            "classifier_tolerance": 0.0001,
            "classifier_max_iter": 1000,
        },
    )
