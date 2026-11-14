from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import load_config
from tests.conftest import REPO, REPO_CONFIG

REPO_CONFIGS = [REPO_CONFIG, *sorted((REPO / "configs").glob("*.yaml"))]


@pytest.mark.parametrize("path", REPO_CONFIGS, ids=lambda p: p.name)
def test_repo_configs_are_valid(path: Path) -> None:
    load_config(path)


def test_extends_merges_sections_but_replaces_model_params() -> None:
    base = load_config(REPO_CONFIG)
    baseline = load_config(REPO / "configs" / "baseline.yaml")

    assert baseline.model.type == "dummy"
    assert baseline.model.params == {}  # replaced, not merged with the forest's params
    assert baseline.data == base.data  # inherited untouched
    assert baseline.features == base.features


def test_set_overrides_parse_values_as_yaml() -> None:
    cfg = load_config(
        REPO_CONFIG,
        ["model.params.max_depth=3", "split.stratify=false", "tracking.tags=[a, b]"],
    )
    assert cfg.model.params["max_depth"] == 3
    assert cfg.split.stratify is False
    assert cfg.tracking.tags == ["a", "b"]


def test_set_requires_key_equals_value() -> None:
    with pytest.raises(ValueError, match=r"key\.path=value"):
        load_config(REPO_CONFIG, ["model.params.max_depth"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("split.test_size=0.9", "leave rows for training"),
        ("evaluation.primary_metric=rmse", "isn't available for classification"),
        ("features.numeric=[churned]", "also listed as a feature"),
        ("features.categorical=[age]", "listed more than once"),
        ("model.type=svm", "model.type"),
        ("data.remote_uri=https://example.org/data.csv", "must be an s3:// or gs:// URI"),
        ("ingest.remote_prefix=bucket/raw", "must be an s3:// or gs:// URI"),
        ("split.time_column=churned", "can't be the split's time_column"),
        ("data.typo=1", "Extra inputs are not permitted"),
    ],
)
def test_invalid_configs_are_rejected(override: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_config(REPO_CONFIG, [override])


def test_extends_loop_is_reported(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="loop"):
        load_config(tmp_path / "a.yaml")


def test_cloud_locations_accept_s3_and_gcs() -> None:
    cfg = load_config(
        REPO_CONFIG,
        [
            "data.remote_uri=gs://bucket/processed/data.parquet",
            "ingest.remote_prefix=gs://bucket/raw",
            "output.remote_prefix=s3://bucket/models",
        ],
    )
    assert cfg.data.remote_uri == "gs://bucket/processed/data.parquet"


def test_sources_parse_with_defaults() -> None:
    cfg = load_config(
        REPO_CONFIG,
        [
            "ingest.sources=[{name: base, type: file, uri: a.csv},"
            " {name: extra, type: http, url: 'https://example.org', join_on: [id]}]"
        ],
    )
    base, extra = cfg.ingest.sources
    assert base.join_on is None
    assert (extra.format, extra.how, extra.join_on) == ("json", "left", ["id"])


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ("[{name: a, type: file}]", "needs a uri"),
        ("[{name: a, type: http}]", "needs a url"),
        ("[{name: Bad-Name, type: file, uri: a.csv}]", "String should match pattern"),
        (
            "[{name: a, type: file, uri: a.csv}, {name: a, type: file, uri: b.csv}]",
            "more than once",
        ),
        ("[{name: a, type: file, uri: a.csv, join_on: [id]}]", "can't have join_on"),
        (
            "[{name: a, type: file, uri: a.csv}, {name: b, type: file, uri: b.csv, join_on: []}]",
            "at least one column",
        ),
        ("[{name: a, type: file, uri: a.csv, how: outer}]", "how"),
    ],
)
def test_invalid_sources_are_rejected(sources: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_config(REPO_CONFIG, [f"ingest.sources={sources}"])
