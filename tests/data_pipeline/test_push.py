from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline import push
from ml_pipeline.demo_data import make_demo_frame
from shared.config import PipelineConfig, load_config
from tests.conftest import REPO_CONFIG

REMOTE = "gs://bucket/processed/data.parquet"


def processed_config(tmp_path: Path, *overrides: str) -> PipelineConfig:
    path = tmp_path / "data.parquet"
    make_demo_frame(n_rows=20).to_parquet(path, index=False)
    return load_config(
        REPO_CONFIG, [f"data.local_path={path}", f"data.remote_uri={REMOTE}", *overrides]
    )


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(push, "upload", lambda src, uri: calls.append((src, uri)) or True)
    return calls


def test_push_uploads_the_processed_table(tmp_path: Path, uploads: list) -> None:
    cfg = processed_config(tmp_path)
    assert push.push(cfg) == REMOTE
    assert uploads == [(cfg.data.local_path, REMOTE)]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["data.remote_uri=null"], "remote_uri isn't set"),
        (["data.local_path=nowhere.parquet"], "make prepare"),
        (["features.numeric=[age, not_a_column]"], "lacks columns"),
    ],
)
def test_push_refuses_when_something_is_wrong(
    tmp_path: Path, uploads: list, overrides: list[str], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        push.push(processed_config(tmp_path, *overrides))
    assert uploads == []


def test_force_pushes_despite_missing_columns(tmp_path: Path, uploads: list) -> None:
    push.push(processed_config(tmp_path, "features.numeric=[not_a_column]"), force=True)
    assert len(uploads) == 1


def test_push_fails_when_the_upload_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(push, "upload", lambda src, uri: False)
    with pytest.raises(SystemExit, match="no Google Cloud credentials"):
        push.push(processed_config(tmp_path))
