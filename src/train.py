"""Train, evaluate, log, and save a model, all driven by a config file.

    python -m src.train --config config.yaml
    python -m src.train --config configs/linear.yaml --set model.params.C=0.5

Each run writes models/<run-id>/ (model.joblib, metadata.json, plots/) and a
copy in models/latest/, logs everything to W&B, and uploads the model files to
output.remote_prefix when it's set and AWS credentials are available.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig, load_config
from src.data import Split, file_sha256, load_dataset, split_dataset
from src.evaluate import compute_metrics, make_plots
from src.features import build_preprocessor
from src.models import build_estimator
from src.storage import join_uri, upload
from src.tracking import log_model, log_results, start_run

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class TrainResult:
    run_id: str
    model_dir: Path
    metrics: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(cfg.features)),
            ("model", build_estimator(cfg.task, cfg.model.type, cfg.model.params, cfg.seed)),
        ]
    )


def train(cfg: PipelineConfig, config_path: str | Path = "config.yaml") -> TrainResult:
    run = start_run(cfg, config_path)
    try:
        result = _train(cfg, run)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
    return result


def _train(cfg: PipelineConfig, run) -> TrainResult:
    frame = load_dataset(cfg)
    split = split_dataset(frame, cfg)
    log.info("Split sizes: %s", split.sizes())

    model = build_pipeline(cfg)
    log.info("Training %s (%s)", cfg.model.type, cfg.task)
    model.fit(split.X_train, split.y_train)

    scores = {
        "val": compute_metrics(cfg.task, model, split.X_val, split.y_val),
        "test": compute_metrics(cfg.task, model, split.X_test, split.y_test),
    }
    primary = cfg.evaluation.primary_metric
    for name in ("val", "test"):
        if primary in scores[name]:
            log.info("%s %s: %.4f", name, primary, scores[name][primary])

    model_dir = cfg.output.local_dir / run.id
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE, compress=3)
    metadata = build_metadata(cfg, run, model, split, scores, frame)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    plots = make_plots(cfg.task, model, split.X_test, split.y_test, model_dir / "plots")

    log_results(run, scores, plots, primary)
    log_model(run, model_dir, metadata)
    _publish(cfg, model_dir, run.id)
    log.info("Saved model %s to %s", run.id, model_dir)
    return TrainResult(run.id, model_dir, scores, metadata)


def build_metadata(
    cfg: PipelineConfig,
    run,
    model: Pipeline,
    split: Split,
    scores: dict[str, dict[str, float]],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Everything the API and front-end need to know about the model, saved next to it."""
    features = [_describe_numeric(split.X_train[c], c) for c in cfg.features.numeric]
    features += [_describe_categorical(split.X_train[c], c) for c in cfg.features.categorical]
    metadata: dict[str, Any] = {
        "model_version": run.id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": cfg.task,
        "target": cfg.data.target,
        "model_type": cfg.model.type,
        "primary_metric": cfg.evaluation.primary_metric,
        "metrics": scores,
        "features": features,
        "example": {f["name"]: f["example"] for f in features},
        "data": {
            "uri": cfg.data.remote_uri or str(cfg.data.local_path),
            "sha256": file_sha256(cfg.data.local_path),
            "rows": len(frame),
            "split": split.sizes(),
        },
        "wandb_url": run.url,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "config": cfg.model_dump(mode="json"),
    }
    if cfg.task == "classification":
        metadata["classes"] = [_to_python(c) for c in model.classes_]
    return metadata


def _describe_numeric(column: pd.Series, name: str) -> dict[str, Any]:
    values = column.dropna()
    median = float(values.median())
    return {
        "name": name,
        "kind": "numeric",
        "min": float(values.min()),
        "max": float(values.max()),
        "example": int(median) if pd.api.types.is_integer_dtype(column) else round(median, 4),
    }


def _describe_categorical(column: pd.Series, name: str) -> dict[str, Any]:
    values = column.dropna()
    return {
        "name": name,
        "kind": "categorical",
        "categories": sorted((_to_python(v) for v in values.unique()), key=str),
        "example": _to_python(values.mode().iloc[0]),
    }


def _to_python(value: Any) -> Any:
    """numpy scalars -> plain Python, so they serialize to JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _publish(cfg: PipelineConfig, model_dir: Path, run_id: str) -> None:
    """Copy the run to models/latest/ locally, and to Cloud Storage when output.remote_prefix is set."""
    latest = cfg.output.local_dir / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(model_dir, latest)
    if not cfg.output.remote_prefix:
        return
    for folder in (run_id, "latest"):
        for name in (MODEL_FILE, METADATA_FILE):
            upload(model_dir / name, join_uri(cfg.output.remote_prefix, folder, name))


def main(argv: list[str] | None = None) -> TrainResult:
    parser = argparse.ArgumentParser(description="Train a model from a config file.")
    parser.add_argument(
        "--config", default="config.yaml", help="config file (default: config.yaml)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set model.params.max_depth=5 (repeatable)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    return train(cfg, args.config)


if __name__ == "__main__":
    main()
