"""Data pipeline, stage 2 of 3: clean each raw snapshot, join them, and add engineered features.

    python -m data_pipeline.prepare --config config.yaml

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

The result is written to `data.local_path`, ready for stage 3 (data_pipeline/push.py).

Where features belong: the API gets one row of inputs from the front-end, so a
column computed here has to be something the front-end can supply or look up
too. Transforms of a row's own inputs (ratios, logs, date parts) are better
left to the ML pipeline's saved model, which the API applies for you, along
with model-time transforms (imputing, scaling, encoding).

Snapshots are CSV, so codes with leading zeros (state FIPS "01", ZIP codes)
come back as numbers. Restore them in the source's cleaner, e.g.
`frame["fips"] = frame["fips"].astype(str).str.zfill(2)`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from data_pipeline.ingest import snapshot_file
from shared.cli import config_parser, load
from shared.config import PipelineConfig, SourceConfig
from shared.storage import download, join_uri
from shared.tables import missing_columns, write_table

log = logging.getLogger(__name__)

Cleaner = Callable[[pd.DataFrame], pd.DataFrame]
FeatureStep = Callable[..., pd.DataFrame]  # (table, sources, **options) -> table
CLEANERS: dict[str, Cleaner] = {}
FEATURE_STEPS: dict[str, FeatureStep] = {}


def cleaner(source_name: str) -> Callable[[Cleaner], Cleaner]:
    """Register `fn(frame) -> frame` as the cleaner for one source. It runs after standard_clean."""

    def register(fn: Cleaner) -> Cleaner:
        CLEANERS[source_name] = fn
        return fn

    return register


def feature_step(name: str) -> Callable[[FeatureStep], FeatureStep]:
    """Register `fn(table, sources, **options) -> table` as a feature step.

    `table` is the joined table. `sources` maps each source name to its cleaned
    frame, including sources left out of the join. `options` come from
    `prepare.step_options.<name>` in the config, so a step's settings live there.
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
        options = cfg.prepare.step_options.get(name, {})
        table = _checked(FEATURE_STEPS[name](table, frames, **options), f"feature step {name}")
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
            f"feature steps {unknown} aren't registered in data_pipeline/prepare.py. "
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


# Dataset-specific cleaners and feature steps go below.
#
# The weather dataset (config.yaml): one row per North Carolina airport per
# day, with today's and yesterday's temperatures as features, and the label
# `tomorrow_category`, whether tomorrow's high is expected, an extreme high,
# or an extreme low for that station and time of year.

EXPECTED, EXTREME_HIGH, EXTREME_LOW = "expected", "extreme high", "extreme low"
WEATHER_COLUMNS = [
    "date",
    "station_id",
    "station",
    "latitude",
    "longitude",
    "elevation_m",
    "month",
    "today_high_f",
    "today_low_f",
    "yesterday_high_f",
    "tomorrow_high_f",
    "today_category",
    "tomorrow_category",
]


@cleaner("weather")
def clean_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """NOAA's one row per reading, as one row per station and day with its high and low.

    Readings come as (date, datatype TMAX or TMIN, station, attributes, value).
    `attributes` holds NOAA's flags, "measurement,quality,source,time". A
    quality flag means the reading failed one of NOAA's checks, so it's dropped.
    """
    flags = frame["attributes"].fillna("").astype(str).str.split(",")
    failed = flags.str[1].fillna("").str.strip() != ""
    if failed.any():
        log.info("weather: dropping %d readings that failed NOAA's quality checks", failed.sum())
    readings = frame.loc[~failed]
    daily = (
        readings.pivot_table(
            index=["station", "date"], columns="datatype", values="value", aggfunc="first"
        )
        .rename(columns={"TMAX": "high_f", "TMIN": "low_f"})
        .reset_index()
        .rename(columns={"station": "station_id"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily[["station_id", "date", "high_f", "low_f"]]


@cleaner("stations")
def clean_stations(frame: pd.DataFrame) -> pd.DataFrame:
    """Each station's id, a short name for the form, and where it is.

    "RALEIGH DURHAM INTERNATIONAL AIRPORT, NC US" becomes "Raleigh Durham".
    """
    names = (
        frame["name"]
        .str.replace(r",\s*[A-Z]{2}\s+US$", "", regex=True)
        .str.replace(r"\s+(?:INTERNATIONAL\s+|REGIONAL\s+)?AIRPORT$", "", regex=True)
        .str.title()
    )
    return pd.DataFrame(
        {
            "station_id": frame["id"],
            "station": names,
            "latitude": frame["latitude"],
            "longitude": frame["longitude"],
            "elevation_m": frame["elevation"],
        }
    )


@feature_step("daily_lags")
def daily_lags(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Add yesterday's and tomorrow's highs, matched by calendar date rather than row order.

    So a day a station didn't report leaves a gap (missing) instead of pairing
    the wrong days. Renames high_f and low_f to today_high_f and today_low_f.
    """
    highs = table[["station_id", "date", "high_f"]]
    one_day = pd.Timedelta(days=1)
    yesterday = highs.assign(date=highs["date"] + one_day).rename(
        columns={"high_f": "yesterday_high_f"}
    )
    tomorrow = highs.assign(date=highs["date"] - one_day).rename(
        columns={"high_f": "tomorrow_high_f"}
    )
    return (
        table.merge(yesterday, on=["station_id", "date"], how="left")
        .merge(tomorrow, on=["station_id", "date"], how="left")
        .rename(columns={"high_f": "today_high_f", "low_f": "today_low_f"})
    )


@feature_step("tomorrow_label")
def tomorrow_label(
    table: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    *,
    baseline_years: tuple[int, int] = (1991, 2020),
    window_days: int = 7,
    percentiles: tuple[float, float] = (10, 90),
) -> pd.DataFrame:
    """Label today's and tomorrow's highs against the station's normal range for the date.

    The normal range for a station and calendar day runs from the low to the
    high `percentiles` of its daily highs in `baseline_years`, using the days
    within `window_days` of that date in each year. Above it is "extreme
    high", below it "extreme low", and in between "expected". A fixed baseline
    keeps later (test) years from shaping their own labels.

    Adds `month`, `today_category` (a persistence baseline, not a feature), and
    the target `tomorrow_category`. Drops rows missing a feature or the label.
    """
    first, last = baseline_years
    low_pct, high_pct = percentiles
    today = _calendar_day(table["date"])
    in_baseline = table["date"].dt.year.between(first, last) & table["today_high_f"].notna()
    ranges = _normal_ranges(
        table.loc[in_baseline, "station_id"],
        today[in_baseline],
        table.loc[in_baseline, "today_high_f"],
        window_days=window_days,
        percentiles=(low_pct, high_pct),
    )
    tomorrow = _calendar_day(table["date"] + pd.Timedelta(days=1))
    table = table.assign(
        month=table["date"].dt.month_name(),
        today_category=_categorize(table["today_high_f"], table["station_id"], today, ranges),
        tomorrow_category=_categorize(
            table["tomorrow_high_f"], table["station_id"], tomorrow, ranges
        ),
    )

    required = ["today_high_f", "today_low_f", "yesterday_high_f", "tomorrow_category"]
    complete = table[required].notna().all(axis=1)
    if not complete.all():
        log.info("weather: dropping %d rows missing a feature or the label", (~complete).sum())
    table = table.loc[complete].reset_index(drop=True)
    shares = table["tomorrow_category"].value_counts(normalize=True).round(3).to_dict()
    log.info("weather: tomorrow_category shares %s", shares)
    return table[[c for c in WEATHER_COLUMNS if c in table.columns]]


def _calendar_day(dates: pd.Series) -> pd.Series:
    """Day of the year, 1 to 365, counting Feb 29 as Feb 28 so leap years line up."""
    leap = dates.dt.is_leap_year
    after_feb_28 = leap & ((dates.dt.month > 2) | ((dates.dt.month == 2) & (dates.dt.day == 29)))
    return dates.dt.day_of_year - after_feb_28.astype(int)


def _normal_ranges(
    stations: pd.Series,
    days: pd.Series,
    highs: pd.Series,
    *,
    window_days: int,
    percentiles: tuple[float, float],
) -> pd.DataFrame:
    """The (low, high) percentiles of each station's highs around each calendar day."""
    rows = []
    baseline = pd.DataFrame({"station_id": stations, "day": days, "high": highs})
    for station_id, group in baseline.groupby("station_id"):
        group_days, group_highs = group["day"].to_numpy(), group["high"].to_numpy()
        for day in range(1, 366):
            gap = np.abs(group_days - day)
            gap = np.minimum(gap, 365 - gap)  # Dec 31 is next to Jan 1
            window = group_highs[gap <= window_days]
            low, high = np.percentile(window, percentiles) if window.size else (np.nan, np.nan)
            rows.append((station_id, day, low, high))
    if not rows:
        raise ValueError("no rows in the baseline years to measure the normal range from")
    return pd.DataFrame(rows, columns=["station_id", "day", "low", "high"]).set_index(
        ["station_id", "day"]
    )


def _categorize(
    highs: pd.Series, stations: pd.Series, days: pd.Series, ranges: pd.DataFrame
) -> pd.Series:
    """Each high as expected / extreme high / extreme low, or missing if either side is."""
    bounds = ranges.reindex(pd.MultiIndex.from_arrays([stations, days]))
    low, high = bounds["low"].to_numpy(), bounds["high"].to_numpy()
    values = highs.to_numpy(dtype=float)
    labels = np.where(values > high, EXTREME_HIGH, np.where(values < low, EXTREME_LOW, EXPECTED))
    known = ~(np.isnan(values) | np.isnan(low))
    return pd.Series(labels, index=highs.index, dtype=object).where(known)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = config_parser("Clean, join, and add features to the raw data.")
    return prepare(load(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
