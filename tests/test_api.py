import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import create_app, load_bundle
from src.demo_data import make_demo_frame
from src.train import train
from tests.conftest import REGRESSION, REPO, demo_config, disable_wandb

SAMPLE_REQUEST = json.loads((REPO / "tests" / "sample_request.json").read_text())


@pytest.fixture(scope="module")
def train_model(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    """Train a demo model once per set of overrides and return its model.joblib path."""
    root = tmp_path_factory.mktemp("api")
    data = root / "demo.csv"
    make_demo_frame(n_rows=200).to_csv(data, index=False)
    trained: dict[tuple[str, ...], Path] = {}

    def make(*overrides: str) -> Path:
        if overrides not in trained:
            with pytest.MonkeyPatch.context() as mp:
                disable_wandb(mp, root)
                out = root / f"models-{len(trained)}"
                result = train(demo_config(data, out, *overrides))
            trained[overrides] = result.model_dir / "model.joblib"
        return trained[overrides]

    return make


@pytest.fixture(scope="module")
def client(train_model: Callable[..., Path]) -> TestClient:
    return TestClient(create_app(str(train_model())))


def test_health_reports_model_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == client.get("/metadata").json()["model_version"]


def test_metadata_describes_features(client: TestClient) -> None:
    features = {f["name"]: f for f in client.get("/metadata").json()["features"]}
    assert set(features) == set(SAMPLE_REQUEST)
    assert features["region"]["categories"] == ["east", "north", "south", "west"]


def test_predict_sample_request(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_REQUEST)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] in ("no", "yes")
    assert 0.5 <= body["probability"] <= 1  # probability of the predicted class


@pytest.mark.parametrize(
    "change",
    [
        {"region": "mars"},  # category not seen in training
        {"age": "old"},  # not a number
        {"extra_field": 1},  # unknown feature
    ],
    ids=["unknown-category", "wrong-type", "extra-field"],
)
def test_invalid_requests_return_422(client: TestClient, change: dict) -> None:
    assert client.post("/predict", json={**SAMPLE_REQUEST, **change}).status_code == 422


def test_missing_feature_returns_422(client: TestClient) -> None:
    request = {k: v for k, v in SAMPLE_REQUEST.items() if k != "plan"}
    assert client.post("/predict", json=request).status_code == 422


def test_swagger_schema_lists_categories_and_an_example(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]["PredictRequest"]
    assert schema["properties"]["plan"]["enum"] == ["basic", "premium", "standard"]
    assert set(schema["examples"][0]) == set(SAMPLE_REQUEST)


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.headers["location"] == "/docs"


def test_regression_model_has_no_probability(train_model: Callable[..., Path]) -> None:
    client = TestClient(create_app(str(train_model(*REGRESSION))))
    body = client.post("/predict", json=SAMPLE_REQUEST).json()
    assert isinstance(body["prediction"], float)
    assert body["probability"] is None


def test_missing_model_says_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        create_app(str(tmp_path / "model.joblib"))


def test_s3_model_downloads_both_files_to_the_cache(
    train_model: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = train_model().parent
    requested: list[str] = []

    def fake_download(uri: str, dest: Path) -> Path:
        requested.append(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_dir / uri.rsplit("/", 1)[1], dest)
        return dest

    monkeypatch.setattr(api.main, "download", fake_download)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    bundle = load_bundle("s3://bucket/models/latest/model.joblib")

    assert requested == [
        "s3://bucket/models/latest/model.joblib",
        "s3://bucket/models/latest/metadata.json",
    ]
    assert bundle.feature_names == list(SAMPLE_REQUEST)
