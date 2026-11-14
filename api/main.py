"""FastAPI app that serves a trained model pipeline.

Run it with the app factory, since the request schema depends on the model:

    uvicorn api.main:create_app --factory --port 8080

MODEL_URI picks the model: a local path (default: models/latest/model.joblib)
or an s3:// or gs:// URI. metadata.json is read from the same folder. Cloud files are
downloaded to MODEL_CACHE_DIR (default: a folder under /tmp, the only writable
path on Lambda).

This module deliberately doesn't use `from __future__ import annotations`:
FastAPI reads the /predict parameter's type at runtime, and that type is built
inside create_app().
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.schemas import HealthResponse, MetadataResponse, PredictResponse, build_request_model
from src.storage import download, is_remote_uri, join_uri

log = logging.getLogger(__name__)

DEFAULT_MODEL_URI = "models/latest/model.joblib"


@dataclass
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.metadata["features"]]


def load_bundle(model_uri: str) -> ModelBundle:
    """Load model.joblib and the metadata.json next to it, from disk or cloud storage."""
    metadata_uri = join_uri(model_uri.rsplit("/", 1)[0], "metadata.json")
    if is_remote_uri(model_uri):
        cache = Path(os.environ.get("MODEL_CACHE_DIR", Path(tempfile.gettempdir()) / "model-cache"))
        model_path = download(model_uri, cache / "model.joblib")
        metadata_path = download(metadata_uri, cache / "metadata.json")
    else:
        model_path, metadata_path = Path(model_uri), Path(metadata_uri)
        if not model_path.exists():
            raise FileNotFoundError(
                f"no model at {model_path}. Train one first (make train) or set MODEL_URI."
            )

    metadata = json.loads(metadata_path.read_text())
    trained_with = metadata.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        log.warning(
            "Model was trained with scikit-learn %s but %s is installed; predictions may fail",
            trained_with,
            sklearn.__version__,
        )
    return ModelBundle(joblib.load(model_path), metadata)


def create_app(model_uri: str | None = None) -> FastAPI:
    model_uri = model_uri or os.environ.get("MODEL_URI") or DEFAULT_MODEL_URI
    bundle = load_bundle(model_uri)
    log.info("Serving model %s from %s", bundle.version, model_uri)
    is_classifier = bundle.metadata["task"] == "classification"

    PredictRequest = build_request_model(bundle.metadata["features"], bundle.metadata["example"])

    app = FastAPI(
        title="Prediction API",
        description=(
            f"Predicts **{bundle.metadata['target']}** ({bundle.metadata['task']}) with a "
            f"{bundle.metadata['model_type']} model. See GET /metadata for its features."
        ),
        version=bundle.version,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(model_version=bundle.version)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> dict[str, Any]:
        """The model's features (with allowed values and ranges), classes, and test metrics."""
        return bundle.metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        row = pd.DataFrame([request.model_dump()], columns=bundle.feature_names)
        prediction = bundle.pipeline.predict(row)[0]
        probability = None
        if is_classifier and hasattr(bundle.pipeline, "predict_proba"):
            probability = float(bundle.pipeline.predict_proba(row)[0].max())
        return PredictResponse(
            prediction=prediction.item() if hasattr(prediction, "item") else prediction,
            probability=probability,
            model_version=bundle.version,
        )

    return app
