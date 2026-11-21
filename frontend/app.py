"""Streamlit front-end: a form for the model's inputs that asks the deployed API for a prediction.

    API_URL=http://localhost:8080 streamlit run frontend/app.py

API_URL is the API's base URL: an environment variable locally, or a secret on
Streamlit Community Cloud. The form is built from the API's GET /metadata, so
it always matches the model the API serves.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from api_client import ApiError, describe, get_metadata, label, ordered, predict

st.set_page_config(page_title="Is tomorrow normal?", page_icon="🌡️", layout="centered")


def api_url() -> str | None:
    if os.environ.get("API_URL"):
        return os.environ["API_URL"]
    try:
        return st.secrets.get("API_URL")
    except Exception:  # no secrets file at all
        return None


@st.cache_data(ttl=300, show_spinner="Connecting to the model…")
def metadata(base_url: str) -> dict[str, Any]:
    return get_metadata(base_url)


def feature_input(feature: dict[str, Any]) -> Any:
    name, example = feature["name"], feature.get("example")
    if feature["kind"] == "categorical":
        options = ordered(feature["categories"])
        index = options.index(example) if example in options else 0
        return st.selectbox(label(name), options, index=index)
    low, high = feature.get("min"), feature.get("max")
    default = example if example is not None else (low + high) / 2
    hint = f"The model saw {low:g} to {high:g} in training." if low is not None else None
    return st.number_input(label(name), value=float(default), step=1.0, format="%.1f", help=hint)


st.title("🌡️ Is tomorrow's weather normal?")
st.write(
    "Enter today's weather at a North Carolina station. The model predicts whether "
    "**tomorrow's high** will be *expected*, an *extreme high*, or an *extreme low* "
    "for that place and time of year."
)

base_url = api_url()
if not base_url:
    st.error(
        "API_URL isn't set. Set it as an environment variable, or as a secret on Streamlit Cloud."
    )
    st.stop()

try:
    meta = metadata(base_url)
except ApiError as err:
    st.error(str(err))
    if st.button("Try again"):
        st.rerun()
    st.stop()

with st.form("inputs"):
    columns = st.columns(2)
    inputs = {}
    for i, feature in enumerate(meta["features"]):
        with columns[i % 2]:
            inputs[feature["name"]] = feature_input(feature)
    submitted = st.form_submit_button("Predict tomorrow", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Asking the model…"):
        try:
            result = predict(base_url, inputs)
        except ApiError as err:
            st.error(str(err))
            st.stop()
    icon, headline, sentence = describe(result["prediction"])
    st.subheader(f"{icon} {headline}")
    if sentence:
        st.write(sentence)
    if result.get("probability") is not None:
        st.progress(result["probability"], text=f"Model confidence: {result['probability']:.0%}")
    st.caption(f"Model version `{result['model_version']}`")

with st.expander("About this model"):
    metric = meta.get("primary_metric")
    test_score = meta.get("metrics", {}).get("test", {}).get(metric)
    st.write(f"**Predicts:** `{meta['target']}` with a {meta['model_type']} model.")
    if test_score is not None:
        st.write(f"**Test {metric}:** {test_score:.3f}")
    st.write(
        "A class project, not a forecast service. For weather warnings, check the "
        "[National Weather Service](https://www.weather.gov)."
    )
