from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import requests

from scripts.ingest_data import main
from src import ingest
from src.config import SourceConfig
from tests.conftest import REPO_CONFIG, pipeline_config

SECRET = "not-a-real-key-1234"


class FakeResponse:
    def __init__(self, payload: Any = None, *, text: str = "", status: int = 200) -> None:
        self.payload, self.text, self.status_code = payload, text, status

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        return self.payload


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the network: map URLs to FakeResponses (or exceptions) and record each request."""
    fake = SimpleNamespace(responses={}, calls=[])

    def get(session: requests.Session, url: str, **kwargs: Any) -> FakeResponse:
        fake.calls.append({"url": url, **kwargs})
        response = fake.responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    return fake


@pytest.fixture
def table_csv(tmp_path: Path) -> Path:
    path = tmp_path / "table.csv"
    pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]}).to_csv(path, index=False)
    return path


def test_file_source_is_saved_with_a_manifest(tmp_path: Path, table_csv: Path) -> None:
    cfg = pipeline_config(tmp_path, [{"name": "table", "type": "file", "uri": str(table_csv)}])

    manifest = ingest.ingest(cfg)

    snapshot = pd.read_csv(tmp_path / "raw" / "table.csv")
    assert snapshot["value"].tolist() == [10.0, 20.0, 30.0]
    entry = manifest["table"]
    assert (entry["rows"], entry["columns"], entry["location"]) == (
        3,
        ["id", "value"],
        str(table_csv),
    )
    assert len(entry["sha256"]) == 64
    assert json.loads((tmp_path / "raw" / "manifest.json").read_text()) == manifest


def test_http_json_source_reads_records_and_expands_env_vars(
    tmp_path: Path, http: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_API_KEY", SECRET)
    http.responses["https://api.example.org/v1"] = FakeResponse(
        {"results": {"items": [{"state": "NC", "stats": {"loss": 0.2}}]}}
    )
    source = {
        "name": "api",
        "type": "http",
        "url": "https://api.example.org/v1",
        "params": {"key": "${TEST_API_KEY}", "year": 2024},
        "records": "results.items",
    }

    manifest = ingest.ingest(pipeline_config(tmp_path, [source]))

    (call,) = http.calls
    assert call["params"] == {"key": SECRET, "year": 2024}
    assert "User-Agent" in call["headers"]
    snapshot = pd.read_csv(tmp_path / "raw" / "api.csv")
    assert snapshot.to_dict("records") == [{"state": "NC", "stats.loss": 0.2}]
    # The manifest keeps the placeholder, never the key itself.
    assert manifest["api"]["params"]["key"] == "${TEST_API_KEY}"
    assert SECRET not in (tmp_path / "raw" / "manifest.json").read_text()


def test_http_csv_source(tmp_path: Path, http: SimpleNamespace) -> None:
    http.responses["https://example.org/data.csv"] = FakeResponse(text="a,b\n1,x\n2,y\n")
    source = {"name": "csv", "type": "http", "url": "https://example.org/data.csv", "format": "csv"}

    ingest.ingest(pipeline_config(tmp_path, [source]))

    assert pd.read_csv(tmp_path / "raw" / "csv.csv")["b"].tolist() == ["x", "y"]


def test_missing_env_var_is_named(tmp_path: Path, http: SimpleNamespace) -> None:
    source = {
        "name": "api",
        "type": "http",
        "url": "https://x.org",
        "params": {"key": "${NOPE_KEY}"},
    }
    with pytest.raises(ValueError, match="needs the environment variable NOPE_KEY"):
        ingest.ingest(pipeline_config(tmp_path, [source]))


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(text=f"invalid key {SECRET}", status=403),
        requests.ConnectionError(f"Max retries exceeded with url: /v1?key={SECRET}"),
    ],
    ids=["http-error", "network-error"],
)
def test_errors_never_show_api_keys(
    tmp_path: Path, http: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, response: Any
) -> None:
    monkeypatch.setenv("TEST_API_KEY", SECRET)
    http.responses["https://api.example.org/v1"] = response
    source = {
        "name": "api",
        "type": "http",
        "url": "https://api.example.org/v1",
        "params": {"key": "${TEST_API_KEY}"},
    }

    with pytest.raises(ConnectionError) as caught:
        ingest.ingest(pipeline_config(tmp_path, [source]))

    assert SECRET not in str(caught.value)
    assert "***" in str(caught.value)
    assert caught.value.__cause__ is None  # the original message held the key


def test_records_path_must_lead_to_a_list(tmp_path: Path, http: SimpleNamespace) -> None:
    http.responses["https://x.org"] = FakeResponse({"data": {"not": "a list"}})
    source = {"name": "api", "type": "http", "url": "https://x.org", "records": "data"}
    with pytest.raises(ValueError, match="expected a list of records at 'data'"):
        ingest.ingest(pipeline_config(tmp_path, [source]))


def test_unknown_type_fails_before_anything_is_fetched(tmp_path: Path, table_csv: Path) -> None:
    sources = [
        {"name": "table", "type": "file", "uri": str(table_csv)},
        {"name": "other", "type": "no_such_type"},
    ]
    with pytest.raises(ValueError, match="isn't registered"):
        ingest.ingest(pipeline_config(tmp_path, sources))
    assert not (tmp_path / "raw" / "table.csv").exists()


def test_custom_source_types_can_be_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch_fake(source: SourceConfig) -> pd.DataFrame:
        return pd.DataFrame({"n": range(source.options["rows"])})

    monkeypatch.setitem(ingest.FETCHERS, "fake", fetch_fake)
    cfg = pipeline_config(tmp_path, [{"name": "made", "type": "fake", "options": {"rows": 4}}])

    assert ingest.ingest(cfg)["made"]["rows"] == 4


def test_only_refetches_the_named_sources(tmp_path: Path, table_csv: Path) -> None:
    sources = [
        {"name": "first", "type": "file", "uri": str(table_csv)},
        {"name": "second", "type": "file", "uri": str(table_csv), "join_on": ["id"]},
    ]
    cfg = pipeline_config(tmp_path, sources)
    ingest.ingest(cfg)
    (tmp_path / "raw" / "second.csv").unlink()

    manifest = ingest.ingest(cfg, only=["first"])

    assert set(manifest) == {"first", "second"}  # the earlier entry is kept
    assert not (tmp_path / "raw" / "second.csv").exists()
    with pytest.raises(ValueError, match="no source named"):
        ingest.ingest(cfg, only=["third"])


def test_snapshots_are_uploaded_when_remote_prefix_is_set(
    tmp_path: Path, table_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[str] = []
    monkeypatch.setattr(ingest, "upload", lambda src, uri: uploaded.append(uri) or True)
    cfg = pipeline_config(
        tmp_path,
        [{"name": "table", "type": "file", "uri": str(table_csv)}],
        "ingest.remote_prefix=gs://bucket/raw",
    )

    ingest.ingest(cfg)

    assert uploaded == ["gs://bucket/raw/table.csv", "gs://bucket/raw/manifest.json"]


def test_no_sources_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sources to ingest"):
        ingest.ingest(pipeline_config(tmp_path, []))


def test_command_line(tmp_path: Path, table_csv: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"extends: {REPO_CONFIG}\n"
        f"ingest:\n  raw_dir: {tmp_path / 'raw'}\n"
        f"  sources:\n    - {{name: table, type: file, uri: {table_csv}}}\n"
    )

    manifest = main(["--config", str(config), "--source", "table"])

    assert manifest["table"]["rows"] == 3


def test_http_json_columns_become_rows(tmp_path: Path, http: SimpleNamespace) -> None:
    # Open-Meteo style: one list per column under `daily`.
    http.responses["https://x.org"] = FakeResponse(
        {"daily": {"time": ["2024-01-01", "2024-01-02"], "sunshine_duration": [3600.0, 0.0]}}
    )
    source = {"name": "sun", "type": "http", "url": "https://x.org", "records": "daily"}

    ingest.ingest(pipeline_config(tmp_path, [source]))

    snapshot = pd.read_csv(tmp_path / "raw" / "sun.csv")
    assert snapshot.to_dict("list") == {
        "time": ["2024-01-01", "2024-01-02"],
        "sunshine_duration": [3600.0, 0.0],
    }
