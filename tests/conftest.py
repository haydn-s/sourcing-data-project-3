"""Shared fixtures: synthetic data and configs that write only to temp folders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.config import PipelineConfig, load_config
from src.demo_data import make_demo_frame

REPO = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO / "config.yaml"


def disable_wandb(monkeypatch: pytest.MonkeyPatch, wandb_dir: Path) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never log to W&B or write a ./wandb folder."""
    disable_wandb(monkeypatch, tmp_path)


@pytest.fixture
def demo_csv(tmp_path: Path) -> Path:
    path = tmp_path / "demo.csv"
    make_demo_frame(n_rows=200).to_csv(path, index=False)
    return path


def demo_config(data_path: Path, out_dir: Path, *overrides: str) -> PipelineConfig:
    """The repo's config.yaml, pointed at temp folders, with a small fast forest."""
    return load_config(
        REPO_CONFIG,
        [
            f"data.local_path={data_path}",
            f"output.local_dir={out_dir}",
            "model.params.n_estimators=20",
            *overrides,
        ],
    )


@pytest.fixture
def config_for(demo_csv: Path, tmp_path: Path) -> Callable[..., PipelineConfig]:
    """Build a demo config with extra `key=value` overrides: config_for("task=regression")."""

    def make(*overrides: str) -> PipelineConfig:
        return demo_config(demo_csv, tmp_path / "models", *overrides)

    return make


REGRESSION = (
    "task=regression",
    "data.target=lifetime_value",
    "split.stratify=false",
    "evaluation.primary_metric=rmse",
)
