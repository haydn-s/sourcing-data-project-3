from __future__ import annotations

import pandas as pd
import pytest

from ml_pipeline.data import split_dataset
from ml_pipeline.demo_data import make_demo_frame
from shared.config import load_config
from shared.tables import missing_columns
from tests.conftest import REPO_CONFIG


def quarterly_frame(n_periods: int = 20, rows_per_period: int = 5) -> pd.DataFrame:
    """Demo rows tagged with a period, several rows per period (like states in a quarter)."""
    frame = make_demo_frame(n_rows=n_periods * rows_per_period)
    periods = [p for p in range(n_periods) for _ in range(rows_per_period)]
    # Shuffle so the split can't rely on the rows already being in time order.
    return frame.assign(period=periods).sample(frac=1, random_state=0).reset_index(drop=True)


def test_time_split_keeps_periods_together_and_in_order() -> None:
    cfg = load_config(REPO_CONFIG, ["split.time_column=period"])
    frame = quarterly_frame()

    split = split_dataset(frame, cfg)

    period = frame["period"]
    train, val, test = (period[part.index] for part in (split.X_train, split.X_val, split.X_test))
    assert train.max() < val.min() <= val.max() < test.min()
    assert sum(split.sizes().values()) == len(frame)
    assert set(train) | set(val) | set(test) == set(range(20))  # no period is split up
    assert list(split.X_train.columns) == cfg.features.all  # the time column isn't a feature


def test_time_split_needs_enough_periods() -> None:
    cfg = load_config(REPO_CONFIG, ["split.time_column=period"])
    with pytest.raises(ValueError, match="too few distinct values"):
        split_dataset(quarterly_frame(n_periods=2, rows_per_period=10), cfg)


def test_time_split_rejects_missing_times() -> None:
    cfg = load_config(REPO_CONFIG, ["split.time_column=period"])
    frame = quarterly_frame()
    frame.loc[0, "period"] = None
    with pytest.raises(ValueError, match="missing values"):
        split_dataset(frame, cfg)


def test_missing_columns_includes_the_time_column() -> None:
    cfg = load_config(REPO_CONFIG, ["split.time_column=period"])
    assert missing_columns(make_demo_frame(n_rows=5), cfg) == ["period"]
