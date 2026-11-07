"""Load the processed dataset and split it into train, validation, and test sets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PipelineConfig
from src.storage import download

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


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file (local path or URL), chosen by extension."""
    suffix = Path(str(path).split("?")[0]).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported file type '{suffix}' for {path}: use .csv or .parquet")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported file type '{path.suffix}' for {path}: use .csv or .parquet")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_columns(frame: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    return [c for c in (*cfg.features.all, cfg.data.target) if c not in frame.columns]


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
    """Seeded train/validation/test split, stratified for classification when configured."""
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
