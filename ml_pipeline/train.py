"""ML pipeline, stage 4 of 4: split, fit, evaluate, save, and publish."""

from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ml_pipeline.evaluate import evaluate
from ml_pipeline.models import build_model
from ml_pipeline.pull import pull_training_data
from shared.cli import config_parser, load
from shared.config import PipelineConfig
from shared.storage import join_uri, upload

log = logging.getLogger(__name__)


def _split_data(
    X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    stratify = y if cfg.task == "classification" and cfg.split.stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )
    validation_fraction = cfg.split.val_size / (1 - cfg.split.test_size)
    stratify_train_val = (
        y_train_val if cfg.task == "classification" and cfg.split.stratify else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=cfg.seed,
        stratify=stratify_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _json_scalar(value: Any, *, context: str) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{context} must contain finite JSON scalar values; got {value!r}")


def _feature_metadata(
    X: pd.DataFrame, cfg: PipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[dict[str, Any]] = []
    example: dict[str, Any] = {}

    for name in cfg.features.numeric:
        values = pd.to_numeric(X[name], errors="raise").dropna()
        if values.empty:
            raise ValueError(f"numeric feature '{name}' has no observed training values")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"numeric feature '{name}' contains an infinite training value")
        minimum, maximum = float(numeric.min()), float(numeric.max())
        features.append(
            {
                "name": name,
                "kind": "numeric",
                "min": minimum,
                "max": maximum,
                "example": float(numeric[0]),
            }
        )
        example[name] = float(numeric[0])

    for name in cfg.features.categorical:
        values = X[name].dropna()
        categories = [
            _json_scalar(value, context=f"categorical feature '{name}'")
            for value in pd.unique(values)
        ]
        if not categories:
            raise ValueError(f"categorical feature '{name}' has no observed training values")
        features.append(
            {
                "name": name,
                "kind": "categorical",
                "categories": categories,
                "example": categories[0],
            }
        )
        example[name] = categories[0]

    return features, example


def _save_artifacts(
    cfg: PipelineConfig,
    model: Any,
    X_train: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> tuple[str, Path]:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"  # noqa: UP017
    run_dir = cfg.output.local_dir / run_id

    features, example = _feature_metadata(X_train, cfg)
    classes = None
    if cfg.task == "classification":
        classes = [_json_scalar(value, context="classification target") for value in model.classes_]
    metadata = {
        "model_version": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "task": cfg.task,
        "target": cfg.data.target,
        "model_type": cfg.model.type,
        "primary_metric": cfg.evaluation.primary_metric,
        "metrics": metrics,
        "features": features,
        "classes": classes,
        "example": example,
        "versions": {"scikit-learn": sklearn.__version__},
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "model.joblib"
    metadata_path = run_dir / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    latest_dir = cfg.output.local_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, latest_dir / model_path.name)
    shutil.copyfile(metadata_path, latest_dir / metadata_path.name)

    if cfg.output.remote_prefix:
        for version in (run_id, "latest"):
            for artifact in (model_path, metadata_path):
                uri = join_uri(cfg.output.remote_prefix, version, artifact.name)
                if not upload(artifact, uri):
                    raise RuntimeError(
                        f"could not publish {artifact.name} to {uri}; "
                        "check Google Cloud credentials and write access"
                    )

    return run_id, run_dir


def _track_run(cfg: PipelineConfig, metrics: dict[str, dict[str, float]]) -> None:
    import wandb

    run = wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        tags=cfg.tracking.tags,
        config={
            "seed": cfg.seed,
            "task": cfg.task,
            "target": cfg.data.target,
            "features": cfg.features.all,
            "model_type": cfg.model.type,
            "model_params": cfg.model.params,
            "primary_metric": cfg.evaluation.primary_metric,
        },
    )
    if run is None:
        raise RuntimeError("Weights & Biases did not create a run")
    try:
        run.log(
            {
                f"{split}/{metric}": score
                for split, scores in metrics.items()
                for metric, score in scores.items()
            }
        )
    finally:
        run.finish()


def train(cfg: PipelineConfig) -> Path:
    """Run training and return the saved run directory."""
    X, y = pull_training_data(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y, cfg)

    model = build_model(cfg)
    model.fit(X_train, y_train)
    metrics = {
        "val": evaluate(model, X_val, y_val, cfg),
        "test": evaluate(model, X_test, y_test, cfg),
    }

    run_id, run_dir = _save_artifacts(cfg, model, X_train, metrics)
    _track_run(cfg, metrics)
    log.info(
        "Trained %s model %s; %s=%s (val), %s (test)",
        cfg.model.type,
        run_id,
        cfg.evaluation.primary_metric,
        metrics["val"][cfg.evaluation.primary_metric],
        metrics["test"][cfg.evaluation.primary_metric],
    )
    log.info("Saved model and metadata to %s and %s", run_dir, cfg.output.local_dir / "latest")
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = config_parser("Train, evaluate, and publish the configured model.")
    return train(load(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
