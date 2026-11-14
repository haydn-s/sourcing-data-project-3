"""Data pipeline, stage 1 of 3: fetch the raw data from every source in the config.

    python -m scripts.ingest_data --config config.yaml
    python -m scripts.ingest_data --source colonies     # just one source

Saves each source's rows unchanged to ingest.raw_dir as <name>.csv, plus a
manifest.json, and uploads them to ingest.remote_prefix when it's set. Sources
and their types are described in src/ingest.py and config.yaml.

Next: python -m scripts.prepare_data
"""

from __future__ import annotations

from typing import Any

from scripts._cli import config_parser, load
from src.ingest import ingest


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = config_parser("Fetch the raw data from every source in the config.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        metavar="NAME",
        help="fetch only this source (repeatable). Default: all of them",
    )
    args = parser.parse_args(argv)
    return ingest(load(args), only=args.sources)


if __name__ == "__main__":
    main()
