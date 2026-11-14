from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline import prepare
from data_pipeline.ingest import ingest
from ml_pipeline.demo_data import make_demo_frame
from shared.config import SourceConfig
from tests.conftest import pipeline_config


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("AverageTemperature", "average_temperature"),
        ("State Name", "state_name"),
        ("CV (%)", "cv"),
        ("PM2.5", "pm2_5"),
        ("already_snake", "already_snake"),
    ],
)
def test_snake_case(raw: str, clean: str) -> None:
    assert prepare.snake_case(raw) == clean


def test_standard_clean() -> None:
    raw = pd.DataFrame(
        {
            "State Name": ["  North Carolina ", "Ohio", "", "Ohio", None],
            "Value": [1.0, 2.0, np.nan, 2.0, np.nan],
            "Empty": [None] * 5,
            "value": [5, 6, 7, 6, 8],  # becomes a second "value"
        }
    )

    clean = prepare.standard_clean(raw)

    assert list(clean.columns) == ["state_name", "value", "value_2"]  # empty column dropped
    assert clean["state_name"].tolist()[:2] == ["North Carolina", "Ohio"]
    assert clean["state_name"].isna().sum() == 2  # the blank and the None
    assert len(clean) == 4  # one duplicate row dropped


def frame(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def source(name: str, join_on: list[str] | None = None, how: str = "left") -> SourceConfig:
    return SourceConfig(name=name, type="file", uri=f"{name}.csv", join_on=join_on, how=how)


def test_left_join_suffixes_overlapping_columns_and_reports_unmatched_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    left = frame(state=["NC", "OH", "TX"], value=[1, 2, 3])
    right = frame(state=["NC", "OH"], value=[10, 20], temp=[15.0, 10.0])

    with caplog.at_level(logging.WARNING):
        joined = prepare.join_one(left, right, source("climate", ["state"]))

    assert list(joined.columns) == ["state", "value", "value_climate", "temp"]
    assert joined["temp"].tolist()[:2] == [15.0, 10.0]
    assert np.isnan(joined["temp"].iloc[2])
    assert "1 of 3 rows found no match" in caplog.text


def test_inner_join_drops_unmatched_rows() -> None:
    left = frame(state=["NC", "OH", "TX"])
    right = frame(state=["NC"], temp=[15.0])
    joined = prepare.join_one(left, right, source("climate", ["state"], how="inner"))
    assert joined["state"].tolist() == ["NC"]


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (frame(state=["NC", "NC"], temp=[1.0, 2.0]), "repeat a"),
        (frame(region=["NC"], temp=[1.0]), "has no column"),
        (frame(state=[1], temp=[1.0]), "int64 in the source"),
    ],
    ids=["repeated-keys", "missing-key", "key-types-differ"],
)
def test_joins_that_would_go_wrong_are_rejected(right: pd.DataFrame, message: str) -> None:
    left = frame(state=["NC", "OH"])
    with pytest.raises(ValueError, match=message):
        prepare.join_one(left, right, source("climate", ["state"]))


def test_blank_join_keys_never_match() -> None:
    left = frame(state=["NC", None])
    right = frame(state=["NC", None], temp=[15.0, 99.0])

    joined = prepare.join_one(left, right, source("climate", ["state"]))

    assert joined["temp"].tolist()[0] == 15.0
    assert np.isnan(joined["temp"].iloc[1])  # not the 99.0 from the other blank-key row


def write_csv(path: Path, data: pd.DataFrame) -> str:
    data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def two_sources(tmp_path: Path) -> list[dict]:
    """The demo churn data split into customers plus a region lookup table."""
    customers = make_demo_frame(n_rows=60)
    regions = pd.DataFrame(
        {"Region": ["north", "south", "east", "west"], "Avg Income": [50, 40, 60, 55]}
    )
    return [
        {"name": "customers", "type": "file", "uri": write_csv(tmp_path / "c.csv", customers)},
        {
            "name": "regions",
            "type": "file",
            "uri": write_csv(tmp_path / "r.csv", regions),
            "join_on": ["region"],
        },
    ]


def test_cleaners_and_feature_steps_run_in_order(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def upper_regions(data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(region=data["region"].str.upper())

    def spend_share(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return table.assign(spend_share=table["monthly_spend"] / table["avg_income"])

    # Both sources get uppercased, so the join still matches.
    monkeypatch.setitem(prepare.CLEANERS, "customers", upper_regions)
    monkeypatch.setitem(prepare.CLEANERS, "regions", upper_regions)
    monkeypatch.setitem(prepare.FEATURE_STEPS, "spend_share", spend_share)
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[spend_share]")
    ingest(cfg)

    table = prepare.prepare(cfg)

    assert set(table["region"]) <= {"NORTH", "SOUTH", "EAST", "WEST"}
    assert table["avg_income"].notna().all()
    assert table["spend_share"].notna().all()
    assert pd.read_parquet(cfg.data.local_path).shape == table.shape


def test_sources_without_join_on_reach_feature_steps_only(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    del two_sources[1]["join_on"]
    seen: dict[str, pd.DataFrame] = {}

    def look(table: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        seen.update(sources)
        return table

    monkeypatch.setitem(prepare.FEATURE_STEPS, "look", look)
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[look]")
    ingest(cfg)

    table = prepare.prepare(cfg)

    assert "avg_income" not in table.columns
    assert list(seen["regions"].columns) == ["region", "avg_income"]


def test_unknown_feature_step_fails_before_any_work(
    tmp_path: Path, two_sources: list[dict]
) -> None:
    cfg = pipeline_config(tmp_path, two_sources, "prepare.feature_steps=[nope]")
    with pytest.raises(ValueError, match="aren't registered"):
        prepare.prepare(cfg)


def test_missing_snapshot_says_to_ingest(tmp_path: Path, two_sources: list[dict]) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        prepare.prepare(pipeline_config(tmp_path, two_sources))


def test_missing_snapshot_is_downloaded_from_the_remote_prefix(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "bucket"
    staged.mkdir()
    ingest(pipeline_config(staged, two_sources))  # snapshots in staged/raw

    def fake_download(uri: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((staged / "raw" / uri.rsplit("/", 1)[1]).read_bytes())
        return dest

    monkeypatch.setattr(prepare, "download", fake_download)
    cfg = pipeline_config(tmp_path, two_sources, "ingest.remote_prefix=gs://bucket/raw")

    assert len(prepare.prepare(cfg)) == 60


def test_command_line_writes_the_processed_table(tmp_path: Path, two_sources: list[dict]) -> None:
    cfg = pipeline_config(tmp_path, two_sources)
    ingest(cfg)
    config = tmp_path / "config.yaml"
    config.write_text(cfg.model_dump_json())  # JSON is valid YAML

    table = prepare.main(["--config", str(config)])

    assert len(table) == 60
    assert cfg.data.local_path.exists()


def test_cleaner_must_return_a_frame(
    tmp_path: Path, two_sources: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forgets_to_return(data: pd.DataFrame) -> None:
        data["region"] = data["region"].str.upper()

    monkeypatch.setitem(prepare.CLEANERS, "customers", forgets_to_return)
    cfg = pipeline_config(tmp_path, two_sources)
    ingest(cfg)

    with pytest.raises(TypeError, match="the cleaner for customers returned NoneType"):
        prepare.prepare(cfg)


def test_empty_snapshot_names_the_source(tmp_path: Path, two_sources: list[dict]) -> None:
    cfg = pipeline_config(tmp_path, two_sources)
    cfg.ingest.raw_dir.mkdir(parents=True)
    (cfg.ingest.raw_dir / "customers.csv").write_text("")

    with pytest.raises(ValueError, match=r"snapshot of source 'customers'.*is empty"):
        prepare.prepare(cfg)
