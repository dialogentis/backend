"""Runtime-facing NLU interface; persistence and trusted loading are separate."""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from dialogentis.datasets.schema import Dataset, Entity
from dialogentis.nlu.entities import EntityPrediction, extract_entities
from dialogentis.nlu.entity_evaluation import EntityEvaluation, evaluate_entities
from dialogentis.nlu.intent import IntentModel, IntentPrediction, Thresholds, train_intents
from dialogentis.nlu.intent_evaluation import IntentEvaluation, evaluate_intents
from dialogentis.nlu.learned_entities import LearnedEntityModel, train_learned_entities


@dataclass(frozen=True)
class NLUResult:
    version_id: UUID
    intent: IntentPrediction
    entities: EntityPrediction


@dataclass(frozen=True)
class NLUReport:
    version_id: UUID
    intents: IntentEvaluation
    entities: EntityEvaluation


@dataclass
class NLUAdapter:
    version_id: UUID
    thresholds: Thresholds
    entities: list[Entity]
    intent_model: IntentModel
    entity_model: LearnedEntityModel | None

    @classmethod
    def train(
        cls,
        version_id: UUID,
        dataset: Dataset,
        thresholds: Thresholds,
        on_stage: Callable[[str], None] | None = None,
    ) -> NLUAdapter:
        snapshot = dataset.model_copy(deep=True)
        if on_stage:
            on_stage("intent_training")
        intent_model = train_intents(snapshot)
        if on_stage and any(item.method == "learned" for item in snapshot.entities):
            on_stage("entity_training")
        entity_model = train_learned_entities(snapshot)
        return cls(
            version_id=version_id,
            thresholds=thresholds,
            entities=snapshot.entities,
            intent_model=intent_model,
            entity_model=entity_model,
        )

    def predict(self, text: str) -> NLUResult:
        return NLUResult(
            self.version_id,
            self.intent_model.predict(text, self.thresholds),
            extract_entities(self.entities, self.entity_model, text),
        )

    def evaluate(self, dataset: Dataset) -> NLUReport:
        # Evaluation must use the entity definitions stored with this model.
        if len(dataset.entities) != len(self.entities) or {
            item.id: item for item in dataset.entities
        } != {item.id: item for item in self.entities}:
            raise ValueError("The test dataset does not match the model's entity definitions")
        return NLUReport(
            self.version_id,
            evaluate_intents(self.intent_model, dataset, self.thresholds),
            evaluate_entities(dataset, self.entity_model),
        )
