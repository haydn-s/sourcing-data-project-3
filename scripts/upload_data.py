"""Clean the raw dataset and upload the processed file (run once, or after cleaning changes).

    python -m scripts.upload_data --config config.yaml --raw <path-or-url>

Reads the raw data (a local CSV/Parquet path, an http(s) URL, or an s3:// URI),
cleans it, writes it to data.local_path, and uploads it to data.remote_uri when
that's set. Training then downloads it from there.

Cleaning is dataset-specific, so clean() is a placeholder until we pick a
dataset. Model-time transforms (imputing, scaling, encoding) don't belong here:
they live in src/features.py so the API applies them too.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.data import missing_columns, read_table, write_table
from src.storage import download, is_s3_uri, upload

log = logging.getLogger(__name__)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw dataset into the processed one. Replace this once we pick a dataset."""
    # TODO(dataset): drop unused or leaky columns, fix types, standardize category
    # labels, handle impossible values and outliers, and document each choice in the
    # README's "Cleaning & preprocessing" section.
    cleaned = raw.drop_duplicates().reset_index(drop=True)
    log.info("Dropped %d duplicate rows", len(raw) - len(cleaned))
    return cleaned


def read_raw(source: str) -> pd.DataFrame:
    if is_s3_uri(source):
        with tempfile.TemporaryDirectory() as tmp:
            return read_table(download(source, Path(tmp) / Path(source).name))
    return read_table(source)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean the raw dataset and upload it.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw", required=True, help="raw data: local path, http(s) URL, or s3://")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    processed = clean(read_raw(args.raw))
    missing = missing_columns(processed, cfg)
    if missing:
        log.warning("The processed data lacks columns the config expects: %s", missing)

    write_table(processed, cfg.data.local_path)
    log.info("Wrote %d rows to %s", len(processed), cfg.data.local_path)
    if cfg.data.remote_uri:
        upload(cfg.data.local_path, cfg.data.remote_uri)
    else:
        log.info("data.remote_uri isn't set, so the processed file stays local")


if __name__ == "__main__":
    main()
