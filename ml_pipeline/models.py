"""ML pipeline, stage 2 of 4: build configured preprocessing and estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from shared.config import PipelineConfig


class _EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Keep original class labels around XGBoost's integer-only classifier API."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> _EncodedClassifier:
        self.label_encoder_ = LabelEncoder()
        self.estimator.fit(X, self.label_encoder_.fit_transform(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = np.asarray(self.estimator.predict(X), dtype=int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)


def build_model(cfg: PipelineConfig) -> Pipeline:
    """Create a preprocessing/model pipeline from the validated config."""
    classifier = cfg.task == "classification"
    model_type = cfg.model.type
    params = dict(cfg.model.params)

    if model_type == "dummy":
        if classifier:
            params.setdefault("random_state", cfg.seed)
            estimator = DummyClassifier(**params)
        else:
            estimator = DummyRegressor(**params)
    elif model_type == "linear":
        if classifier:
            params.setdefault("max_iter", 1000)
            params.setdefault("random_state", cfg.seed)
            estimator = LogisticRegression(**params)
        else:
            estimator = LinearRegression(**params)
    elif model_type == "random_forest":
        params.setdefault("random_state", cfg.seed)
        estimator_type = RandomForestClassifier if classifier else RandomForestRegressor
        estimator = estimator_type(**params)
    elif model_type == "hist_gradient_boosting":
        params.setdefault("random_state", cfg.seed)
        estimator_type = (
            HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        )
        estimator = estimator_type(**params)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as err:
            raise RuntimeError(
                "model.type is xgboost, but xgboost is not installed. "
                "Install it and pin it in requirements.txt before using this model."
            ) from err
        params.setdefault("random_state", cfg.seed)
        estimator_type = XGBClassifier if classifier else XGBRegressor
        estimator = estimator_type(**params)
        if classifier:
            estimator = _EncodedClassifier(estimator)
    else:
        raise ValueError(f"unsupported model type: {model_type}")

    transformers = []
    if cfg.features.numeric:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, cfg.features.numeric))
    if cfg.features.categorical:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, cfg.features.categorical))

    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessing", preprocessing), ("estimator", estimator)])
