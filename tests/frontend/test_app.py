"""Run the Streamlit app against a fake API. Skipped where Streamlit isn't installed (CI)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from frontend import api_client
from tests.conftest import REPO

APP = str(REPO / "frontend" / "app.py")
METADATA = {
    "target": "tomorrow_category",
    "model_type": "linear",
    "primary_metric": "f1",
    "metrics": {"test": {"f1": 0.512}},
    "features": [
        {"name": "station", "kind": "categorical", "categories": ["Wilmington", "Asheville"]},
        {"name": "month", "kind": "categorical", "categories": ["March", "January"]},
        {"name": "today_high_f", "kind": "numeric", "min": 5.0, "max": 104.0, "example": 71.0},
    ],
}


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.status_code, self.ok, self.payload = 200, True, payload

    def json(self) -> Any:
        return self.payload


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    posted: list[dict[str, Any]] = []

    def request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/metadata"):
            return FakeResponse(METADATA)
        posted.append(kwargs["json"])
        return FakeResponse(
            {"prediction": "extreme high", "probability": 0.64, "model_version": "r1"}
        )

    monkeypatch.setattr(api_client.requests, "request", request)
    monkeypatch.setenv("API_URL", "https://api.example.run.app")
    return posted


def test_form_is_built_from_metadata_and_shows_the_prediction(api: list[dict[str, Any]]) -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception

    station, month = app.selectbox
    assert station.options == ["Asheville", "Wilmington"]
    assert month.options == ["January", "March"]
    assert app.number_input[0].label == "Today high (°F)"

    station.select("Wilmington")
    app.number_input[0].set_value(95.0)
    app.button[0].click().run()

    assert api == [{"station": "Wilmington", "month": "January", "today_high_f": 95.0}]
    assert "Extreme high" in app.subheader[0].value


def test_missing_api_url_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)

    app = AppTest.from_file(APP, default_timeout=30).run()

    assert "API_URL isn't set" in app.error[0].value
