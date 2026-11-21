from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.config import load_config
from tests.conftest import REPO_CONFIG


def test_repo_config_is_valid() -> None:
    load_config(REPO_CONFIG)


def test_extends_merges_sections_but_replaces_model_params(tmp_path: Path) -> None:
    base = load_config(REPO_CONFIG)
    variant_path = tmp_path / "variant.yaml"
    variant_path.write_text(f"extends: {REPO_CONFIG}\nmodel:\n  type: dummy\n  params: {{}}\n")

    variant = load_config(variant_path)

    assert variant.model.type == "dummy"
    assert variant.model.params == {}  # replaced, not merged with the parent's params
    assert variant.data == base.data  # inherited untouched
    assert variant.features == base.features


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
        ("features.numeric=[tomorrow_category]", "also listed as a feature"),
        ("features.categorical=[today_high_f]", "listed more than once"),
        ("model.type=svm", "model.type"),
        ("data.remote_uri=https://example.org/data.csv", "must be a gs:// URI"),
        ("ingest.remote_prefix=bucket/raw", "must be a gs:// URI"),
        ("output.remote_prefix=https://example.org/models", "must be a gs:// URI"),
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


def test_cloud_locations_accept_gcs() -> None:
    cfg = load_config(
        REPO_CONFIG,
        [
            "data.remote_uri=gs://bucket/processed/data.parquet",
            "ingest.remote_prefix=gs://bucket/raw",
            "output.remote_prefix=gs://bucket/models",
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


def test_step_options_must_name_a_configured_step() -> None:
    with pytest.raises(ValidationError, match="aren't in feature_steps"):
        load_config(REPO_CONFIG, ["prepare.step_options={nope: {a: 1}}"])


def test_demo_variant_replaces_the_weather_settings() -> None:
    demo = load_config(REPO_CONFIG.parent / "configs" / "demo.yaml")
    assert demo.data.target == "churned"
    assert demo.prepare.feature_steps == []
    assert demo.prepare.step_options == {}  # replaced, not merged with config.yaml's
    assert [source.name for source in demo.ingest.sources] == ["customers"]
