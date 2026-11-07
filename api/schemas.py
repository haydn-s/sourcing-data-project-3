"""Pydantic models for the API.

The /predict request model isn't written by hand: it's built from the feature
list saved in the model's metadata.json, so the API (and its Swagger docs)
always match the model it serves, whatever dataset we train on. Categorical
features only accept the categories seen in training.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str


class PredictResponse(BaseModel):
    prediction: str | int | float | bool
    probability: float | None = Field(
        None, description="Probability of the predicted class. Classification only."
    )
    model_version: str


class FeatureInfo(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    min: float | None = None
    max: float | None = None
    categories: list[Any] | None = None
    example: Any = None


class MetadataResponse(BaseModel):
    model_version: str
    created_at: str
    task: Literal["classification", "regression"]
    target: str
    model_type: str
    primary_metric: str
    metrics: dict[str, dict[str, float]]
    features: list[FeatureInfo]
    classes: list[Any] | None = None
    example: dict[str, Any]


def build_request_model(features: list[dict[str, Any]], example: dict[str, Any]) -> type[BaseModel]:
    """Create the /predict request model from the features in metadata.json."""
    fields: dict[str, Any] = {}
    for feature in features:
        if feature["kind"] == "categorical":
            annotation: Any = Literal[tuple(feature["categories"])]
            description = "One of the categories seen in training."
        else:
            annotation = float
            description = f"Seen in training: {feature['min']:g} to {feature['max']:g}."
        fields[feature["name"]] = (annotation, Field(..., description=description))
    return create_model(
        "PredictRequest",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [example]}),
        **fields,
    )
