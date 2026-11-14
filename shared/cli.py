"""Command-line plumbing for the pipeline commands: --config, --set, .env, and logging."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from shared.config import PipelineConfig, load_config


def config_parser(description: str) -> argparse.ArgumentParser:
    """A parser with the --config and --set options every pipeline command takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default="config.yaml", help="config file (default: config.yaml)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set data.local_path=data/x.parquet (repeatable)",
    )
    return parser


def load(args: argparse.Namespace) -> PipelineConfig:
    """Load .env, set up logging, and read the config named on the command line."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return load_config(args.config, args.overrides)
