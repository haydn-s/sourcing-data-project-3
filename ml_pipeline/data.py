"""Load the processed dataset and split it into train, validation, and test sets."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from shared.config import PipelineConfig
from shared.storage import download
from shared.tables import missing_columns, read_table

log = logging.getLogger(__name__)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def load_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch the processed data (downloading it first if `data.remote_uri` is set) and check it."""
    path = cfg.data.local_path
    if cfg.data.remote_uri:
        download(cfg.data.remote_uri, path)
    if not path.exists():
        raise FileNotFoundError(
            f"no data at {path}. Set data.remote_uri, or for the demo dataset run: make demo-data"
        )
    frame = read_table(path)

    missing = missing_columns(frame, cfg)
    if missing:
        raise ValueError(f"{path} is missing columns named in the config: {missing}")

    unlabeled = frame[cfg.data.target].isna()
    if unlabeled.any():
        log.warning("Dropping %d rows with no target value", int(unlabeled.sum()))
        frame = frame[~unlabeled].reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(frame), path)
    return frame


def split_dataset(frame: pd.DataFrame, cfg: PipelineConfig) -> Split:
    """Seeded train/validation/test split, stratified for classification when configured.

    With `split.time_column` set, the split goes by time instead (see split_by_time).
    """
    if cfg.split.time_column:
        return split_by_time(frame, cfg)
    X = frame[cfg.features.all]
    y = frame[cfg.data.target]
    stratify = cfg.task == "classification" and cfg.split.stratify

    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.seed,
        stratify=y if stratify else None,
    )
    # val_size is a fraction of the whole dataset, so rescale it to the remainder.
    val_fraction = cfg.split.val_size / (1 - cfg.split.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_fraction,
        random_state=cfg.seed,
        stratify=y_rest if stratify else None,
    )
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def split_by_time(frame: pd.DataFrame, cfg: PipelineConfig) -> Split:
    """The latest rows become the test set, and the ones just before them the validation set.

    A random split would let the model learn from rows that come after the ones
    it's tested on. The cut points fall between time values, so rows from the
    same period (one quarter across every state, say) always land in the same set.
    """
    column = cfg.split.time_column
    times = frame[column]
    if times.isna().any():
        raise ValueError(
            f"time_column '{column}' has {int(times.isna().sum())} missing values. "
            "Drop or fill them when preparing the data"
        )
    ordered = times.sort_values(kind="stable").reset_index(drop=True)
    n = len(ordered)
    test_start = ordered.iloc[min(n - 1, round(n * (1 - cfg.split.test_size)))]
    val_start = ordered.iloc[round(n * (1 - cfg.split.test_size - cfg.split.val_size))]

    is_test = times >= test_start
    is_val = (times >= val_start) & ~is_test
    is_train = ~(is_test | is_val)
    if not (is_train.any() and is_val.any() and is_test.any()):
        raise ValueError(
            f"time_column '{column}' has too few distinct values ({times.nunique()}) "
            "to make separate train, validation, and test periods"
        )
    log.info("Split by %s: validation from %s, test from %s", column, val_start, test_start)

    X = frame[cfg.features.all]
    y = frame[cfg.data.target]
    return Split(X[is_train], X[is_val], X[is_test], y[is_train], y[is_val], y[is_test])
