"""Weights & Biases logging for training runs.

W&B reads WANDB_MODE from the environment: `offline` logs to ./wandb for a later
`wandb sync`, and `disabled` turns logging off entirely. Either lets anyone train
without a W&B account. Disabled runs still get an ID, which names the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb

from src.config import PipelineConfig


def start_run(cfg: PipelineConfig, config_path: str | Path) -> wandb.Run:
    try:
        return wandb.init(
            project=cfg.tracking.project,
            entity=cfg.tracking.entity,
            config={**cfg.model_dump(mode="json"), "config_file": str(config_path)},
            tags=[*cfg.tracking.tags, cfg.task, cfg.model.type],
            job_type="train",
        )
    except wandb.Error as err:
        raise RuntimeError(
            "W&B couldn't start a run. Run `wandb login` or set WANDB_API_KEY, "
            "or set WANDB_MODE=offline (log locally) or WANDB_MODE=disabled (no logging)."
        ) from err


def log_results(
    run: wandb.Run,
    scores: dict[str, dict[str, float]],
    plots: dict[str, Path],
    primary_metric: str,
) -> None:
    flat = {
        f"{split}/{name}": value
        for split, values in scores.items()
        for name, value in values.items()
    }
    run.summary.update(flat)
    run.summary["primary_metric"] = primary_metric
    run.log({f"plots/{name}": wandb.Image(str(path)) for name, path in plots.items()})


def log_model(run: wandb.Run, model_dir: Path, metadata: dict[str, Any]) -> None:
    """Version the model files as a W&B artifact, tagged `latest`."""
    artifact = wandb.Artifact(
        name=f"model-{run.project or 'local'}",
        type="model",
        metadata={k: metadata[k] for k in ("task", "target", "model_type", "metrics")},
    )
    artifact.add_file(str(model_dir / "model.joblib"))
    artifact.add_file(str(model_dir / "metadata.json"))
    run.log_artifact(artifact, aliases=["latest"])
