"""The front-end's API client, with the network faked. Needs only requests, so it runs in CI."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from frontend import api_client
from frontend.api_client import ApiError, describe, get_metadata, label, ordered, predict

URL = "https://api.example.run.app"


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code, self.payload, self.text = status, payload, text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        if self.payload is None:
            raise ValueError("no JSON")
        return self.payload


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record each request, and answer with `http_response` (a FakeResponse or an exception)."""
    calls: list[dict[str, Any]] = []

    def request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"method": method, "url": url, **kwargs})
        if isinstance(http_response.value, Exception):
            raise http_response.value
        return http_response.value

    monkeypatch.setattr(api_client.requests, "request", request)
    return calls


class http_response:  # noqa: N801  set .value in each test
    value: Any = FakeResponse(payload={})


def test_predict_posts_the_inputs(http: list[dict[str, Any]]) -> None:
    http_response.value = FakeResponse(payload={"prediction": "expected", "probability": 0.8})

    result = predict(URL + "/", {"station": "Raleigh", "today_high_f": 71.0})

    (call,) = http
    assert (call["method"], call["url"]) == ("POST", f"{URL}/predict")
    assert call["json"] == {"station": "Raleigh", "today_high_f": 71.0}
    assert call["timeout"] == api_client.TIMEOUT
    assert result["prediction"] == "expected"


def test_metadata_is_a_get(http: list[dict[str, Any]]) -> None:
    http_response.value = FakeResponse(payload={"features": []})

    assert get_metadata(URL) == {"features": []}
    assert (http[0]["method"], http[0]["url"]) == ("GET", f"{URL}/metadata")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (requests.ConnectionError("refused"), "Couldn't reach the API"),
        (requests.Timeout("slow"), "took too long"),
        (FakeResponse(500), "HTTP 500"),
        (FakeResponse(200, payload=None, text="<html>"), "didn't return JSON"),
        (
            FakeResponse(
                422,
                payload={
                    "detail": [{"loc": ["body", "station"], "msg": "Input should be 'Raleigh'"}]
                },
            ),
            "station: Input should be 'Raleigh'",
        ),
    ],
)
def test_errors_become_readable_messages(
    http: list[dict[str, Any]], response: Any, message: str
) -> None:
    http_response.value = response

    with pytest.raises(ApiError, match=message):
        predict(URL, {})


@pytest.mark.parametrize(
    ("name", "text"),
    [("today_high_f", "Today high (°F)"), ("month", "Month"), ("temp_c", "Temp (°C)")],
)
def test_labels(name: str, text: str) -> None:
    assert label(name) == text


def test_months_are_in_calendar_order_and_other_categories_sorted() -> None:
    assert ordered(["March", "January", "December"]) == ["January", "March", "December"]
    assert ordered(["Wilmington", "Asheville"]) == ["Asheville", "Wilmington"]


def test_known_and_unknown_classes() -> None:
    assert describe("extreme high")[:2] == ("🔥", "Extreme high")
    assert describe("something else") == ("🔮", "something else", "")
