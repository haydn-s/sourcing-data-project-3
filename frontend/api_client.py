"""Talk to the prediction API, and turn its answers into text for the app.

Kept apart from app.py and free of Streamlit, so it's simple to test. It uses
only requests, so the front-end deploys with just frontend/requirements.txt.
"""

from __future__ import annotations

import calendar
from typing import Any

import requests

# Connect, then read. Reads get longer because a Cloud Run service that has
# scaled to zero starts up (and loads the model) on the first request.
TIMEOUT = (5, 60)

MONTHS = list(calendar.month_name)[1:]

# How the app presents each class the model can predict. Any other class is shown as-is.
OUTCOMES = {
    "expected": (
        "✅",
        "Expected",
        "Tomorrow's high should be in the usual range for this place and time of year.",
    ),
    "extreme high": (
        "🔥",
        "Extreme high",
        "Tomorrow's high could be hotter than 9 in 10 years on this date.",
    ),
    "extreme low": (
        "🧊",
        "Extreme low",
        "Tomorrow's high could be colder than 9 in 10 years on this date.",
    ),
}

UNITS = {"_f": " (°F)", "_c": " (°C)"}


class ApiError(Exception):
    """A problem reaching or using the API, with a message to show the user."""


def get_metadata(base_url: str) -> dict[str, Any]:
    """GET /metadata: the model's features, classes, and metrics."""
    return _request("GET", base_url, "/metadata")


def predict(base_url: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """POST /predict with one value per feature. Returns the prediction and its probability."""
    return _request("POST", base_url, "/predict", json=inputs)


def _request(method: str, base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.Timeout:
        raise ApiError(
            "The API took too long to answer. It may be starting up: try again in a minute."
        ) from None
    except requests.RequestException:
        raise ApiError(f"Couldn't reach the API at {base_url}. Is it running?") from None

    if response.status_code == 422:
        raise ApiError(f"The API rejected these inputs: {_validation_detail(response)}")
    if not response.ok:
        raise ApiError(f"The API returned an error (HTTP {response.status_code}). Try again later.")
    try:
        return response.json()
    except ValueError:
        raise ApiError(f"{url} didn't return JSON. Is API_URL the API's address?") from None


def _validation_detail(response: requests.Response) -> str:
    """FastAPI's 422 body, as 'field: problem' pairs."""
    try:
        errors = response.json()["detail"]
        return "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in errors)
    except (ValueError, KeyError, TypeError, IndexError):
        return response.text[:200]


def label(name: str) -> str:
    """A feature name as a form label: `today_high_f` -> `Today high (°F)`."""
    unit = next((text for suffix, text in UNITS.items() if name.endswith(suffix)), "")
    if unit:
        name = name[: -len("_f")]
    return name.replace("_", " ").strip().capitalize() + unit


def ordered(categories: list[Any]) -> list[Any]:
    """Categories in a sensible order for a dropdown: calendar order for month names, else sorted."""
    if categories and all(value in MONTHS for value in categories):
        return sorted(categories, key=MONTHS.index)
    return sorted(categories, key=str)


def describe(prediction: Any) -> tuple[str, str, str]:
    """An icon, a headline, and a sentence for a predicted class."""
    return OUTCOMES.get(str(prediction).lower(), ("🔮", str(prediction), ""))
