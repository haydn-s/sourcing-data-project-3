"""Metrics and diagnostic plots for a fitted model pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline

from shared.config import Task

log = logging.getLogger(__name__)

TOP_FEATURES = 20


def compute_metrics(task: Task, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predicted = model.predict(X)
    if task == "regression":
        return {
            "rmse": float(metrics.root_mean_squared_error(y, predicted)),
            "mae": float(metrics.mean_absolute_error(y, predicted)),
            "r2": float(metrics.r2_score(y, predicted)),
        }

    scores = {
        "accuracy": float(metrics.accuracy_score(y, predicted)),
        "f1": float(metrics.f1_score(y, predicted, average="macro")),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        try:
            if proba.shape[1] == 2:
                # The positive class is the second label in sorted order (e.g. "yes" over "no").
                scores["roc_auc"] = float(
                    metrics.roc_auc_score(y == model.classes_[1], proba[:, 1])
                )
            else:
                scores["roc_auc"] = float(
                    metrics.roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_)
                )
        except ValueError as err:  # e.g. a class missing from this split
            log.warning("Skipping ROC-AUC: %s", err)
    return scores


def make_plots(task: Task, model: Pipeline, X: pd.DataFrame, y: pd.Series, out_dir: Path) -> dict:
    """Save diagnostic plots for the given split as PNGs. Returns {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    if task == "classification":
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics.ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, colorbar=False)
        ax.set_title("Confusion matrix (test set)")
        plots["confusion_matrix"] = _save(fig, out_dir / "confusion_matrix.png")
    else:
        predicted = model.predict(X)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y, predicted, s=12, alpha=0.6)
        lims = [min(y.min(), predicted.min()), max(y.max(), predicted.max())]
        ax.plot(lims, lims, linestyle="--", linewidth=1, color="gray")
        ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs. actual (test set)")
        plots["predicted_vs_actual"] = _save(fig, out_dir / "predicted_vs_actual.png")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(predicted, y - predicted, s=12, alpha=0.6)
        ax.axhline(0, linestyle="--", linewidth=1, color="gray")
        ax.set(xlabel="Predicted", ylabel="Residual (actual - predicted)", title="Residuals")
        plots["residuals"] = _save(fig, out_dir / "residuals.png")

    importance = feature_importance(model)
    if importance is not None:
        top = importance.head(TOP_FEATURES).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.3 * len(top))))
        ax.barh(top.index, top.to_numpy())
        ax.set(xlabel="Importance", title="Feature importance")
        plots["feature_importance"] = _save(fig, out_dir / "feature_importance.png")
    return plots


def feature_importance(model: Pipeline) -> pd.Series | None:
    """Importance per encoded feature, largest first, or None if the model doesn't expose it."""
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        # Coefficients are comparable because numeric features are standardized.
        values = np.abs(np.atleast_2d(estimator.coef_)).mean(axis=0)
    else:
        return None
    names = model.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(values, index=names).sort_values(ascending=False)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
