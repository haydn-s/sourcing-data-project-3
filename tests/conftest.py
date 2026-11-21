"""Shared fixtures: small synthetic tables and configs that write only to temp folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from shared import storage
from shared.config import PipelineConfig, load_config

REPO = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO / "config.yaml"


@pytest.fixture(autouse=True)
def _no_cloud_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the real bucket, even on a machine that's signed in to Google Cloud.

    config.yaml names our bucket, so without this a test could upload to it.
    Tests of storage itself swap in a fake client.
    """

    def blocked(*, signed: bool) -> None:
        raise AssertionError(
            "a test tried to reach Cloud Storage. Use a fake (tests/shared/test_storage.py)"
        )

    monkeypatch.setattr(storage, "has_gcp_credentials", lambda: False)
    monkeypatch.setattr(storage, "_client", blocked)


def sample_table(n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """A small made-up table with the feature and target columns config.yaml names.

    Shaped like the weather dataset (README > Dataset), with random values.
    """
    rng = np.random.default_rng(seed)
    high = rng.normal(70, 15, n_rows).round()
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
            "station": rng.choice(["Asheville", "Raleigh Durham", "Wilmington"], n_rows),
            "month": rng.choice(["January", "April", "July", "October"], n_rows),
            "today_high_f": high,
            "today_low_f": (high - rng.uniform(5, 25, n_rows)).round(),
            "yesterday_high_f": (high + rng.normal(0, 6, n_rows)).round(),
            "tomorrow_category": rng.choice(["expected", "extreme high", "extreme low"], n_rows),
        }
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
            "prepare.feature_steps=[]",  # and to feature steps, since the weather ones
            "prepare.step_options={}",  # need weather data
            f"data.local_path={tmp_path / 'processed' / 'data.parquet'}",
            f"ingest.sources={json.dumps(sources)}",  # JSON is valid YAML
            *overrides,
        ],
    )
