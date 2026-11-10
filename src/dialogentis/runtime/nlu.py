"""Adapt the NLU contract to the runtime without coupling the core to model libraries."""

from collections.abc import Callable

from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.runtime.core import Prediction


def nlu_predictor(adapter: NLUAdapter) -> Callable[[str], Prediction]:
    def predict(text: str) -> Prediction:
        result = adapter.predict(text)
        return Prediction(
            result.version_id, result.intent.intent_id, result.intent.fallback_reasons
        )

    return predict
