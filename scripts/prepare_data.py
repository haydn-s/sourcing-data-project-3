"""Data pipeline, stage 2 of 3: clean the raw snapshots, join them, and add engineered features.

    python -m scripts.prepare_data --config config.yaml

Reads the snapshots that scripts/ingest_data.py saved (downloading them from
ingest.remote_prefix if they aren't local) and writes the processed table to
data.local_path. The cleaning and feature code lives in src/prepare.py.
Model-time transforms (imputing, scaling, encoding) don't belong here: they're
in src/features.py, so the API applies them too.

Next: python -m scripts.push_data
"""

from __future__ import annotations

import pandas as pd

from scripts._cli import config_parser, load
from src.prepare import prepare


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = config_parser("Clean, join, and add features to the raw data.")
    return prepare(load(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
