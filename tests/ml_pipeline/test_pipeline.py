from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from api.schemas import MetadataResponse, build_request_model
from ml_pipeline import evaluate, pull, train
from ml_pipeline.models import _EncodedClassifier
from shared.config import load_config
from tests.conftest import REPO_CONFIG, sample_table


def config(tmp_path: Path, *overrides: str):
    return load_config(
        REPO_CONFIG,
        [
            f"data.local_path={tmp_path / 'processed.parquet'}",
            *overrides,
        ],
    )


def test_pull_downloads_and_selects_configured_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    table = sample_table()
    table.to_parquet(source, index=False)
    cfg = config(tmp_path, "data.remote_uri=gs://bucket/processed/data.parquet")

    def fake_download(uri: str, dest: Path) -> Path:
        assert uri == cfg.data.remote_uri
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return dest

    monkeypatch.setattr(pull, "download", fake_download)

    X, y = pull.pull_training_data(cfg)

    assert X.columns.tolist() == cfg.features.all
    assert y.name == cfg.data.target
    pd.testing.assert_series_equal(y, table[cfg.data.target])


def test_pull_requires_a_remote_uri(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"data\.remote_uri"):
        pull.pull_training_data(config(tmp_path))


def test_evaluate_uses_the_configured_regression_metric(tmp_path: Path) -> None:
    cfg = config(tmp_path, "task=regression", "evaluation.primary_metric=rmse")
    model = Pipeline([("estimator", DummyRegressor(strategy="mean"))])
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model.fit(X, y)

    assert evaluate.evaluate(model, X, y, cfg) == {"rmse": pytest.approx(0.8164965809)}


def test_encoded_classifier_preserves_original_class_labels() -> None:
    model = _EncodedClassifier(DummyClassifier(strategy="most_frequent"))
    X = pd.DataFrame({"value": [1, 2, 3, 4]})
    y = pd.Series(["yes", "no", "yes", "no"])

    model.fit(X, y)

    assert model.classes_.tolist() == ["no", "yes"]
    assert set(model.predict(X)) <= set(y)
    assert model.predict_proba(X).shape == (4, 2)


def test_train_saves_api_compatible_artifacts_and_publishes_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(
        tmp_path,
        f"output.local_dir={tmp_path / 'models'}",
        "output.remote_prefix=gs://bucket/models",
        "evaluation.primary_metric=accuracy",
        'model.params={"n_estimators": 10, "max_depth": 4}',
    )
    table = sample_table(n_rows=120, seed=4)
    X = table[cfg.features.all]
    y = table[cfg.data.target]
    monkeypatch.setattr(train, "pull_training_data", lambda _cfg: (X, y))
    monkeypatch.setattr(train, "_track_run", lambda _cfg, _metrics: None)
    uploads: list[tuple[Path, str]] = []
    monkeypatch.setattr(train, "upload", lambda path, uri: uploads.append((path, uri)) or True)

    run_dir = train.train(cfg)

    model_path = run_dir / "model.joblib"
    metadata_path = run_dir / "metadata.json"
    assert model_path.is_file()
    assert metadata_path.is_file()
    assert joblib.load(model_path).predict(X.iloc[:2]).shape == (2,)

    metadata = json.loads(metadata_path.read_text())
    MetadataResponse.model_validate(metadata)
    request_model = build_request_model(metadata["features"], metadata["example"])
    request_model(**metadata["example"])
    assert set(metadata["metrics"]) == {"val", "test"}
    assert all("accuracy" in split for split in metadata["metrics"].values())
    assert (cfg.output.local_dir / "latest" / "model.joblib").is_file()
    assert len(uploads) == 4
    assert {uri.rsplit("/", 2)[-2] for _, uri in uploads} == {run_dir.name, "latest"}


def test_train_surfaces_failed_cloud_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config(
        tmp_path,
        f"output.local_dir={tmp_path / 'models'}",
        "output.remote_prefix=gs://bucket/models",
        'model.params={"n_estimators": 5}',
    )
    table = sample_table(n_rows=80, seed=5)
    monkeypatch.setattr(
        train, "pull_training_data", lambda _cfg: (table[cfg.features.all], table[cfg.data.target])
    )
    monkeypatch.setattr(train, "_track_run", lambda _cfg, _metrics: None)
    monkeypatch.setattr(train, "upload", lambda _path, _uri: False)

    with pytest.raises(RuntimeError, match="could not publish"):
        train.train(cfg)
