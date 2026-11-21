"""The weather dataset's cleaners and feature steps (data_pipeline/prepare.py), on made-up data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline import prepare
from data_pipeline.ingest import ingest
from data_pipeline.prepare import WEATHER_COLUMNS, clean_stations, clean_weather, daily_lags
from shared.config import load_config
from shared.tables import missing_columns
from tests.conftest import REPO_CONFIG, pipeline_config

WEATHER_STEPS = "prepare.feature_steps=[daily_lags, tomorrow_label]"


def noaa_readings(station: str, dates: pd.DatetimeIndex, highs, lows) -> pd.DataFrame:
    """Rows shaped like a NOAA CDO snapshot: one per reading."""
    stamps = [f"{d:%Y-%m-%d}T00:00:00" for d in dates]
    return pd.concat(
        [
            pd.DataFrame(
                {"date": stamps, "datatype": kind, "station": station, "attributes": ",,W,2400"}
            ).assign(value=values)
            for kind, values in (("TMAX", highs), ("TMIN", lows))
        ],
        ignore_index=True,
    )


def daily_table(highs: pd.Series, station_id: str = "GHCND:A") -> pd.DataFrame:
    """A cleaned and joined weather table: one row per day, indexed by date in `highs`."""
    return pd.DataFrame(
        {
            "station_id": station_id,
            "date": highs.index,
            "high_f": highs.to_numpy(dtype=float),
            "low_f": highs.to_numpy(dtype=float) - 10,
            "station": "Raleigh Durham",
            "latitude": 35.9,
            "longitude": -78.8,
            "elevation_m": 120.5,
        }
    )


def test_clean_weather_makes_one_row_per_station_day_and_drops_failed_readings() -> None:
    raw = noaa_readings("GHCND:A", pd.date_range("2024-01-01", periods=2), [50, 52], [30, 31])
    raw = pd.concat(
        [raw, noaa_readings("GHCND:B", pd.date_range("2024-01-01", periods=1), [60], [40])]
    )
    raw.loc[(raw["station"] == "GHCND:A") & (raw["value"] == 52), "attributes"] = ",I,W,2400"

    daily = clean_weather(prepare.standard_clean(raw.reset_index(drop=True)))

    assert daily.columns.tolist() == ["station_id", "date", "high_f", "low_f"]
    assert pd.api.types.is_datetime64_any_dtype(daily["date"])
    a = daily[daily["station_id"] == "GHCND:A"].set_index("date")
    assert a.loc["2024-01-01", ["high_f", "low_f"]].tolist() == [50.0, 30.0]
    assert np.isnan(a.loc["2024-01-02", "high_f"])  # failed its quality check
    assert a.loc["2024-01-02", "low_f"] == 31.0
    assert len(daily) == 3


def test_clean_stations_shortens_names() -> None:
    raw = pd.DataFrame(
        {
            "id": ["GHCND:A", "GHCND:B", "GHCND:C", "GHCND:D"],
            "name": [
                "RALEIGH DURHAM INTERNATIONAL AIRPORT, NC US",
                "CHARLOTTE DOUGLAS AIRPORT, NC US",
                "ASHEVILLE REGIONAL AIRPORT, NC US",
                "GREENSBORO PIEDMONT TRIAD INTERNATIONAL AIRPORT, NC US",
            ],
            "latitude": [35.9, 35.2, 35.4, 36.1],
            "longitude": [-78.8, -81.0, -82.5, -79.9],
            "elevation": [120.5, 222.6, 645.6, 275.0],
        }
    )

    stations = clean_stations(prepare.standard_clean(raw))

    assert stations["station"].tolist() == [
        "Raleigh Durham",
        "Charlotte Douglas",
        "Asheville",
        "Greensboro Piedmont Triad",
    ]
    assert stations.columns.tolist() == [
        "station_id",
        "station",
        "latitude",
        "longitude",
        "elevation_m",
    ]


def test_daily_lags_match_days_by_date_not_row_order() -> None:
    days = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])  # Jan 3 missing
    table = daily_table(pd.Series([50, 52, 58], index=days))

    lagged = daily_lags(table, {}).set_index("date")

    assert lagged["yesterday_high_f"].tolist()[1] == 50.0
    assert np.isnan(lagged.loc["2024-01-04", "yesterday_high_f"])  # not Jan 2's 52
    assert lagged["tomorrow_high_f"].tolist()[0] == 52.0
    assert np.isnan(lagged.loc["2024-01-02", "tomorrow_high_f"])  # not Jan 4's 58
    assert {"today_high_f", "today_low_f"} <= set(lagged.columns)


def thirty_one_years() -> pd.DataFrame:
    """1991-2021 at one station. In 1991-2020 every day's high is 50 + (year - 1991),
    so each calendar day's baseline runs evenly from 50 to 79. 2021 is 65 except July 2-3."""
    days = pd.date_range("1991-01-01", "2021-12-31", freq="D")
    highs = pd.Series(50.0 + (days.year - 1991), index=days)
    highs[days.year == 2021] = 65.0
    highs["2021-07-02"], highs["2021-07-03"] = 100.0, 0.0
    return daily_lags(daily_table(highs), {})


def test_tomorrow_label_against_the_baseline_range() -> None:
    labeled = prepare.tomorrow_label(thirty_one_years(), {}).set_index("date")

    # Each baseline day's range is the 10th to 90th percentile of 50..79: three
    # values (77, 78, 79) sit above it, and three (50, 51, 52) below.
    baseline = labeled.loc[:"2020-12-30"]
    shares = baseline["today_category"].value_counts(normalize=True)
    assert shares["extreme high"] == pytest.approx(0.1, abs=0.001)
    assert shares["extreme low"] == pytest.approx(0.1, abs=0.001)

    assert labeled.loc["2021-07-01", "tomorrow_category"] == "extreme high"  # July 2: 100
    assert labeled.loc["2021-07-02", "tomorrow_category"] == "extreme low"  # July 3: 0
    assert labeled.loc["2021-07-03", "tomorrow_category"] == "expected"  # July 4: 65
    assert labeled.loc["2021-07-03", "today_category"] == "extreme low"


def test_tomorrow_label_handles_leap_days_and_drops_incomplete_rows() -> None:
    labeled = prepare.tomorrow_label(thirty_one_years(), {})

    assert labeled["tomorrow_category"].notna().all()
    assert pd.Timestamp("1992-02-29") in set(labeled["date"])  # labeled with Feb 28's range
    assert labeled["date"].min() == pd.Timestamp("1991-01-02")  # Jan 1 has no yesterday
    assert labeled["date"].max() == pd.Timestamp("2021-12-30")  # Dec 31 has no tomorrow
    assert labeled.columns.tolist() == WEATHER_COLUMNS
    assert labeled.loc[labeled["date"] == "2021-07-01", "month"].item() == "July"


def test_step_options_come_from_the_config(tmp_path: Path) -> None:
    cfg = pipeline_config(
        tmp_path,
        [],
        WEATHER_STEPS,
        "prepare.step_options={tomorrow_label: {percentiles: [25, 75]}}",
    )
    table = daily_table(
        pd.Series(
            50.0 + (pd.date_range("1991-01-01", "2020-12-31").year - 1991),
            index=pd.date_range("1991-01-01", "2020-12-31"),
        )
    )

    labeled = prepare.add_features(cfg, table, {})

    shares = labeled["today_category"].value_counts(normalize=True)
    assert shares["extreme high"] > 0.2  # wider than the default 10th/90th percentiles


def test_prepare_builds_the_table_config_yaml_describes(tmp_path: Path) -> None:
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(1)
    readings = pd.concat(
        [
            noaa_readings(station, days, highs, highs - 15)
            for station, highs in (
                ("GHCND:USW00013722", rng.normal(70, 12, len(days)).round()),
                ("GHCND:USW00013748", rng.normal(74, 9, len(days)).round()),
            )
        ]
    )
    stations = pd.DataFrame(
        {
            "id": ["GHCND:USW00013722", "GHCND:USW00013748"],
            "name": [
                "RALEIGH DURHAM INTERNATIONAL AIRPORT, NC US",
                "WILMINGTON INTERNATIONAL AIRPORT, NC US",
            ],
            "latitude": [35.89, 34.27],
            "longitude": [-78.78, -77.90],
            "elevation": [120.5, 7.0],
            "elevationUnit": ["METERS", "METERS"],
        }
    )
    readings.to_csv(tmp_path / "w.csv", index=False)
    stations.to_csv(tmp_path / "s.csv", index=False)
    sources = [
        {"name": "weather", "type": "file", "uri": str(tmp_path / "w.csv")},
        {
            "name": "stations",
            "type": "file",
            "uri": str(tmp_path / "s.csv"),
            "join_on": ["station_id"],
        },
    ]
    cfg = pipeline_config(
        tmp_path,
        sources,
        WEATHER_STEPS,
        "prepare.step_options={tomorrow_label: {baseline_years: [2023, 2023]}}",
    )
    ingest(cfg)

    table = prepare.prepare(cfg)

    assert table.columns.tolist() == WEATHER_COLUMNS
    assert missing_columns(table, load_config(REPO_CONFIG)) == []  # what make push checks
    assert set(table["station"]) == {"Raleigh Durham", "Wilmington"}
    assert set(table["tomorrow_category"]) == {"expected", "extreme high", "extreme low"}
    assert table[["today_high_f", "yesterday_high_f", "tomorrow_category"]].notna().all().all()
    assert len(table) == 2 * (len(days) - 2)  # every day but the first and last
