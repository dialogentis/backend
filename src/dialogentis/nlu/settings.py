from pydantic import Field, StrictFloat

from dialogentis.contracts import InputModel


class NLUSettings(InputModel):
    score: StrictFloat = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    margin: StrictFloat = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    similarity: StrictFloat = Field(default=0.20, ge=0, le=1, allow_inf_nan=False)
    vocabulary_coverage: StrictFloat = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    classifier_c: StrictFloat = Field(default=1.0, ge=0.01, le=100, allow_inf_nan=False)


class QualitySettings(InputModel):
    intent_f1: StrictFloat = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    entity_f1: StrictFloat = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
