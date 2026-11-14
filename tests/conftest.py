"""Shared fixtures: synthetic data and configs that write only to temp folders."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src import storage
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


@pytest.fixture(autouse=True)
def _no_cloud_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the real bucket, even on a machine that's signed in to Google Cloud.

    config.yaml names our bucket, so without this a test could upload to it.
    Tests of storage itself swap in a fake client.
    """

    def blocked(*, signed: bool) -> None:
        raise AssertionError(
            "a test tried to reach Cloud Storage. Use a fake (tests/test_storage.py)"
        )

    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)
    monkeypatch.setattr(storage, "_client", blocked)


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


def pipeline_config(
    tmp_path: Path, sources: list[dict[str, Any]], *overrides: str
) -> PipelineConfig:
    """The repo's config.yaml with these data sources, reading and writing only temp folders."""
    return load_config(
        REPO_CONFIG,
        [
            f"ingest.raw_dir={tmp_path / 'raw'}",
            "ingest.remote_prefix=null",  # tests opt in to uploads
            f"data.local_path={tmp_path / 'processed' / 'data.parquet'}",
            f"ingest.sources={json.dumps(sources)}",  # JSON is valid YAML
            *overrides,
        ],
    )
