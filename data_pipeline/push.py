"""Data pipeline, stage 3 of 3: upload the processed table to Cloud Storage.

    python -m data_pipeline.push --config config.yaml

Uploads data.local_path (written by stage 2, data_pipeline/prepare.py) to
data.remote_uri in Cloud Storage. That's the hand-off: the ML pipeline
downloads the table from there (ml_pipeline/data.py), and since the bucket is
public, so can anyone else.

It refuses to upload a table that's missing columns the config's features,
target, or split use, since training on it would fail. --force skips that check.
"""

from __future__ import annotations

import logging

from shared.cli import config_parser, load
from shared.config import PipelineConfig
from shared.storage import public_url, upload
from shared.tables import file_sha256, missing_columns, table_columns

log = logging.getLogger(__name__)


def push(cfg: PipelineConfig, *, force: bool = False) -> str:
    """Upload the processed table and return where it went."""
    remote_uri, path = cfg.data.remote_uri, cfg.data.local_path
    if not remote_uri:
        raise SystemExit(
            "data.remote_uri isn't set in the config, so there's nowhere to push to. "
            "Set it to a location like gs://my-bucket/processed/data.parquet"
        )
    if not path.exists():
        raise SystemExit(f"no processed data at {path}. Run: make prepare")

    missing = missing_columns(table_columns(path), cfg)
    if missing and not force:
        raise SystemExit(
            f"{path} lacks columns the config expects: {missing}. Fix the config or "
            "data_pipeline/prepare.py, or pass --force to upload it anyway"
        )
    if not upload(path, remote_uri):
        raise SystemExit(
            f"{path} wasn't uploaded: no Google Cloud credentials (see the warning above)"
        )

    log.info("Pushed %s (sha256 %s)", remote_uri, file_sha256(path))
    log.info("Anyone can read it at %s", public_url(remote_uri))
    return remote_uri


def main(argv: list[str] | None = None) -> str:
    parser = config_parser("Upload the processed data to data.remote_uri.")
    parser.add_argument(
        "--force", action="store_true", help="upload even if config columns are missing"
    )
    args = parser.parse_args(argv)
    return push(load(args), force=args.force)


if __name__ == "__main__":
    main()
