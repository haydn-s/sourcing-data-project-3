"""Stage 2 of the data pipeline: clean each raw snapshot, join them, and add engineered features.

1. Clean. Every source gets `standard_clean`: snake_case column names, trimmed
   text, blank strings as missing, and no empty rows, empty columns, or
   duplicate rows. Then the source's own cleaner runs, if one is registered
   below with `@cleaner("<source name>")`. Cleaners see the snake_case names.
2. Join. The first source in `ingest.sources` is the base table. Each later
   source with `join_on` is merged onto it by those columns (a left join, unless
   it sets `how: inner`). A source without `join_on` stays out of the join, but
   feature steps can still use it, e.g. to compute a distance to the nearest fault.
3. Engineer features. The steps named in `prepare.feature_steps` run in order.
   Register each one below with `@feature_step("<name>")`.

The result is written to `data.local_path`, ready for scripts/push_data.py.

Where features belong: the API gets one row of inputs from the front-end, so a
column computed here has to be something the front-end can supply or look up
too. Transforms of a row's own inputs (ratios, logs, date parts) are better
added to the model pipeline in src/features.py, which the API applies for you.

Snapshots are CSV, so codes with leading zeros (state FIPS "01", ZIP codes)
come back as numbers. Restore them in the source's cleaner, e.g.
`frame["fips"] = frame["fips"].astype(str).str.zfill(2)`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping

import pandas as pd

from src.config import PipelineConfig, SourceConfig
from src.data import missing_columns, write_table
from src.ingest import snapshot_file
from src.storage import download, join_uri

log = logging.getLogger(__name__)

Cleaner = Callable[[pd.DataFrame], pd.DataFrame]
FeatureStep = Callable[[pd.DataFrame, Mapping[str, pd.DataFrame]], pd.DataFrame]
CLEANERS: dict[str, Cleaner] = {}
FEATURE_STEPS: dict[str, FeatureStep] = {}


def cleaner(source_name: str) -> Callable[[Cleaner], Cleaner]:
    """Register `fn(frame) -> frame` as the cleaner for one source. It runs after standard_clean."""

    def register(fn: Cleaner) -> Cleaner:
        CLEANERS[source_name] = fn
        return fn

    return register


def feature_step(name: str) -> Callable[[FeatureStep], FeatureStep]:
    """Register `fn(table, sources) -> table` as a feature step.

    `table` is the joined table. `sources` maps each source name to its cleaned
    frame, including sources left out of the join.
    """

    def register(fn: FeatureStep) -> FeatureStep:
        FEATURE_STEPS[name] = fn
        return fn

    return register


# Cleaning


def snake_case(name: object) -> str:
    """`AverageTemperature` -> `average_temperature`, `CV (%)` -> `cv`, `PM2.5` -> `pm2_5`."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def standard_clean(frame: pd.DataFrame, *, source: str = "data") -> pd.DataFrame:
    """The cleaning every source gets before its own cleaner."""
    frame = frame.copy()
    frame.columns = _unique_names(
        [snake_case(c) or f"column_{i}" for i, c in enumerate(frame.columns)], source
    )

    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_string_dtype(values.dtype):
            trimmed = values.map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[column] = trimmed.mask(trimmed == "")

    empty_columns = [c for c in frame.columns if frame[c].isna().all()]
    if empty_columns:
        log.info("%s: dropping columns with no values: %s", source, empty_columns)
    frame = frame.drop(columns=empty_columns).dropna(how="all")

    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    if len(deduplicated) < len(frame):
        log.info("%s: dropped %d duplicate rows", source, len(frame) - len(deduplicated))
    return deduplicated


def _unique_names(names: list[str], source: str) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    renamed = sorted({name for name, count in seen.items() if count > 1})
    if renamed:
        log.warning(
            "%s: several columns became %s in snake_case, so they got _2, _3, ...", source, renamed
        )
    return unique


def load_snapshot(cfg: PipelineConfig, source: SourceConfig) -> pd.DataFrame:
    """Read a source's raw snapshot, downloading it from ingest.remote_prefix if it isn't local."""
    path = cfg.ingest.raw_dir / snapshot_file(source.name)
    if not path.exists():
        if not cfg.ingest.remote_prefix:
            raise FileNotFoundError(
                f"no snapshot of source '{source.name}' at {path}. Run: make ingest"
            )
        download(join_uri(cfg.ingest.remote_prefix, snapshot_file(source.name)), path)
    try:
        # low_memory=False reads each column's type from the whole file, not chunk by chunk.
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"the snapshot of source '{source.name}' at {path} is empty") from err


def _checked(result: object, what: str) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"{what} returned {type(result).__name__}, not a DataFrame. Is `return` missing?"
        )
    return result


def clean_sources(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    frames = {}
    for source in cfg.ingest.sources:
        frame = standard_clean(load_snapshot(cfg, source), source=source.name)
        custom = CLEANERS.get(source.name)
        if custom is None:
            log.info(
                "%s has no cleaner registered, so it gets only the standard clean", source.name
            )
        else:
            frame = _checked(custom(frame), f"the cleaner for {source.name}")
        log.info("%s: %d rows x %d columns after cleaning", source.name, len(frame), frame.shape[1])
        frames[source.name] = frame
    return frames


# Joining


def join_sources(cfg: PipelineConfig, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    base, *others = cfg.ingest.sources
    table = frames[base.name]
    for source in others:
        if source.join_on is not None:
            table = join_one(table, frames[source.name], source)
    return table


def join_one(left: pd.DataFrame, right: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
    """Merge one source onto the table so far, checking the things that silently break joins."""
    keys = list(source.join_on or [])
    for side, frame in (("the table so far", left), (f"source '{source.name}'", right)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"can't join '{source.name}': {side} has no column {missing}")
    for key in keys:
        if pd.api.types.is_numeric_dtype(left[key]) != pd.api.types.is_numeric_dtype(right[key]):
            raise ValueError(
                f"can't join '{source.name}' on '{key}': it's {left[key].dtype} in the table so "
                f"far but {right[key].dtype} in the source. Convert one of them in a cleaner"
            )
    # pandas matches missing keys to each other, so a blank key would pick up an
    # unrelated row. Those rows can't really match anything, so leave them out.
    blank_keys = right[keys].isna().any(axis=1)
    if blank_keys.any():
        log.warning(
            "%s: leaving out %d rows with a missing %s", source.name, int(blank_keys.sum()), keys
        )
        right = right[~blank_keys]
    repeated = right.duplicated(keys)
    if repeated.any():
        raise ValueError(
            f"source '{source.name}' has {int(repeated.sum())} rows that repeat a {keys} "
            "combination, so joining it would duplicate rows. Reduce it to one row per key "
            "(e.g. aggregate) in its cleaner"
        )
    overlap = sorted((set(left.columns) & set(right.columns)) - set(keys))
    if overlap:
        log.warning(
            "%s also has columns %s, so its copies get the suffix _%s",
            source.name,
            overlap,
            source.name,
        )

    merged = left.merge(
        right,
        on=keys,
        how=source.how,
        suffixes=("", f"_{source.name}"),
        indicator="_matched",
    )
    unmatched = int((merged["_matched"] == "left_only").sum())
    if unmatched:
        log.warning(
            "%d of %d rows found no match in %s on %s. Check that the keys are formatted the same way",
            unmatched,
            len(merged),
            source.name,
            keys,
        )
    if len(merged) < len(left):
        log.info("Inner join with %s dropped %d rows", source.name, len(left) - len(merged))
    return merged.drop(columns="_matched")


# Feature engineering


def add_features(
    cfg: PipelineConfig, table: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    for name in cfg.prepare.feature_steps:
        before = set(table.columns)
        table = _checked(FEATURE_STEPS[name](table, frames), f"feature step {name}")
        log.info(
            "Feature step %s added %s", name, sorted(set(table.columns) - before) or "no columns"
        )
    return table


def prepare(cfg: PipelineConfig) -> pd.DataFrame:
    """Clean, join, and add features, then write the processed table to data.local_path."""
    if not cfg.ingest.sources:
        raise ValueError("no sources to prepare. Add them under ingest.sources in the config")
    unknown = [name for name in cfg.prepare.feature_steps if name not in FEATURE_STEPS]
    if unknown:
        raise ValueError(
            f"feature steps {unknown} aren't registered in src/prepare.py. "
            f"Registered: {sorted(FEATURE_STEPS) or 'none yet'}"
        )

    frames = clean_sources(cfg)
    table = add_features(cfg, join_sources(cfg, frames), frames).reset_index(drop=True)

    missing = missing_columns(table, cfg)
    if missing:
        log.warning("The processed data lacks columns the config expects: %s", missing)
    write_table(table, cfg.data.local_path)
    log.info("Wrote %d rows x %d columns to %s", len(table), table.shape[1], cfg.data.local_path)
    return table


# Dataset-specific cleaners and feature steps go below. For example:
#
# @cleaner("colonies")
# def clean_colonies(frame: pd.DataFrame) -> pd.DataFrame:
#     # USDA marks suppressed values with codes like "(Z)" and "(D)".
#     frame["value"] = pd.to_numeric(frame["value"].str.replace(",", ""), errors="coerce")
#     return frame
#
# @feature_step("temperature_anomaly")
# def temperature_anomaly(table, sources):
#     # Measure against a fixed baseline period, not the whole table, so the
#     # test years don't leak into the training rows.
#     baseline = table[table["year"] <= 2010].groupby("state")["avg_temp"].mean()
#     return table.assign(temp_anomaly=table["avg_temp"] - table["state"].map(baseline))
