from pydantic import Field, StrictFloat

from dialogentis.contracts import InputModel


class NLUSettings(InputModel):
    score: StrictFloat = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    margin: StrictFloat = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    similarity: StrictFloat = Field(default=0.20, ge=0, le=1, allow_inf_nan=False)
