from datetime import datetime

from pydantic import BaseModel, Field


class ExplanationFactor(BaseModel):
    name: str
    value: str
    status: str


class ModelFactorContribution(BaseModel):
    feature_name: str
    raw_value: str
    shap_contribution: float
    direction: str


class TimelineEntry(BaseModel):
    step: str
    status: str
    timestamp: datetime | None = None


class ExplanationResponse(BaseModel):
    application_id: str
    decision: str | None = None
    summary: str
    factors: list[ExplanationFactor] = Field(default_factory=list)
    model_factor_contributions: list[ModelFactorContribution] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    rule_version: str | None = None
    model_version: str | None = None
    generated_at: datetime
