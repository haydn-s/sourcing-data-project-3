import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import ml_pipeline.train
from ml_pipeline.data import load_dataset, split_dataset
from ml_pipeline.train import main, train
from shared.config import PipelineConfig
from tests.conftest import REGRESSION, REPO_CONFIG

ConfigFor = Callable[..., PipelineConfig]


def test_training_writes_model_metadata_and_plots(config_for: ConfigFor) -> None:
    cfg = config_for()
    result = train(cfg)

    for name in ("model.joblib", "metadata.json", "plots/confusion_matrix.png"):
        assert (result.model_dir / name).exists(), name
    assert (cfg.output.local_dir / "latest" / "model.joblib").exists()

    metadata = json.loads((result.model_dir / "metadata.json").read_text())
    assert metadata["model_version"] == result.run_id
    assert metadata["classes"] == ["no", "yes"]
    assert [f["name"] for f in metadata["features"]] == cfg.features.all
    assert set(metadata["example"]) == set(cfg.features.all)
    assert set(metadata["metrics"]["test"]) == {"accuracy", "f1", "roc_auc"}
    assert metadata["data"]["split"]["train"] > metadata["data"]["split"]["test"]


def test_regression_is_a_config_change(config_for: ConfigFor) -> None:
    result = train(config_for(*REGRESSION))

    assert set(result.metrics["test"]) == {"rmse", "mae", "r2"}
    assert (result.model_dir / "plots" / "residuals.png").exists()
    assert "classes" not in result.metadata


def test_baseline_model(config_for: ConfigFor) -> None:
    result = train(config_for("model.type=dummy", "model.params={}"))
    assert result.metrics["test"]["roc_auc"] == pytest.approx(0.5)


def test_split_is_seeded_disjoint_and_complete(config_for: ConfigFor) -> None:
    cfg = config_for()
    frame = load_dataset(cfg)
    first, second = split_dataset(frame, cfg), split_dataset(frame, cfg)

    indexes = [set(s.index) for s in (first.y_train, first.y_val, first.y_test)]
    assert sum(len(i) for i in indexes) == len(frame)
    assert not (indexes[0] & indexes[1] or indexes[0] & indexes[2] or indexes[1] & indexes[2])
    pd.testing.assert_index_equal(first.X_test.index, second.X_test.index)


def test_missing_data_file_says_how_to_fix_it(config_for: ConfigFor, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make demo-data"):
        train(config_for(f"data.local_path={tmp_path / 'nope.csv'}"))


def test_uploads_run_and_latest_when_remote_prefix_is_set(
    config_for: ConfigFor, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[str] = []
    monkeypatch.setattr(
        ml_pipeline.train, "upload", lambda src_path, uri: uploaded.append(uri) or True
    )

    result = train(config_for("output.remote_prefix=gs://bucket/models"))

    assert sorted(uploaded) == sorted(
        f"gs://bucket/models/{folder}/{name}"
        for folder in (result.run_id, "latest")
        for name in ("model.joblib", "metadata.json")
    )


def test_cli_applies_set_overrides(demo_csv: Path, tmp_path: Path) -> None:
    result = main(
        [
            "--config",
            str(REPO_CONFIG),
            "--set",
            f"data.local_path={demo_csv}",
            "--set",
            f"output.local_dir={tmp_path / 'models'}",
            "--set",
            "model.params={n_estimators: 5}",
        ]
    )
    assert result.metadata["config"]["model"]["params"] == {"n_estimators": 5}
