"""Read and write data tables, and check one against the config.

The data pipeline writes the processed table and the ML pipeline reads it, so
both use these. `missing_columns` is the contract between them: the columns
the config's features, target, and split need.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from shared.config import PipelineConfig


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


def table_columns(path: Path) -> list[str]:
    """A CSV or Parquet file's column names, read without loading its rows."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    raise ValueError(f"unsupported file type '{path.suffix}' for {path}: use .csv or .parquet")


def missing_columns(table: pd.DataFrame | Iterable[str], cfg: PipelineConfig) -> list[str]:
    """Columns the config names (features, target, time column) that a table doesn't have.

    `table` is a DataFrame or its column names.
    """
    present = set(table.columns if isinstance(table, pd.DataFrame) else table)
    needed = dict.fromkeys([*cfg.features.all, cfg.data.target, cfg.split.time_column])
    return [c for c in needed if c is not None and c not in present]
