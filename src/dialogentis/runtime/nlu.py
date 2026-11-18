"""Adapt the NLU contract to the runtime without coupling the core to model libraries."""

from collections.abc import Callable

from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.runtime.core import EntityValue, Prediction, RuntimeInput


def nlu_predictor(adapter: NLUAdapter) -> Callable[[RuntimeInput], Prediction]:
    def predict(message: RuntimeInput) -> Prediction:
        result = adapter.predict(message.text)
        entities = tuple(
            EntityValue(item.entity_id, item.start, item.end, item.text, item.source)
            for item in result.entities.matches
        )
        return Prediction(
            result.version_id, result.intent.intent_id, result.intent.fallback_reasons, entities
        )

    return predict
