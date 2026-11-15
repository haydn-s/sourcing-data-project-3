"""ML pipeline, stage 1 of 4: load the configured processed table."""

from __future__ import annotations

import pandas as pd

from shared.config import PipelineConfig
from shared.storage import download
from shared.tables import missing_columns, read_table


def pull_training_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Download the processed table and return its configured features and target."""
    if cfg.data.remote_uri is None:
        raise ValueError("Set data.remote_uri in the config before training")

    path = download(cfg.data.remote_uri, cfg.data.local_path)
    table = read_table(path)

    missing = missing_columns(table, cfg)
    if missing:
        raise ValueError(f"{path} lacks columns required for training: {missing}")
    if table[cfg.data.target].isna().any():
        raise ValueError(f"target column '{cfg.data.target}' contains missing values")

    return table.loc[:, cfg.features.all].copy(), table[cfg.data.target].copy()
