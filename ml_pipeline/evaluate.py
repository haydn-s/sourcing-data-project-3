"""ML pipeline, stage 3 of 4: calculate the configured primary metric."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from shared.config import PipelineConfig


def evaluate(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig
) -> dict[str, float]:
    """Score one split using the configured primary metric."""
    metric = cfg.evaluation.primary_metric
    predictions = model.predict(X)

    if cfg.task == "regression":
        if metric == "rmse":
            score = math.sqrt(mean_squared_error(y, predictions))
        elif metric == "mae":
            score = mean_absolute_error(y, predictions)
        elif metric == "r2":
            score = r2_score(y, predictions)
        else:
            raise ValueError(f"unsupported regression metric: {metric}")
    elif metric == "accuracy":
        score = accuracy_score(y, predictions)
    elif metric == "f1":
        score = f1_score(y, predictions, average="macro", zero_division=0)
    elif metric == "roc_auc":
        probabilities = model.predict_proba(X)
        classes = model.classes_
        if len(classes) == 2:
            score = roc_auc_score(y, probabilities[:, 1])
        else:
            score = roc_auc_score(
                y, probabilities, labels=classes, multi_class="ovr", average="macro"
            )
    else:
        raise ValueError(f"unsupported classification metric: {metric}")

    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"metric '{metric}' produced a non-finite score: {value}")
    return {metric: value}
