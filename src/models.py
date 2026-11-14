"""Map `model.type` in the config to a scikit-learn-compatible estimator.

Each type has a classification and a regression version, so a config can
switch tasks without changing the model name. To add a model, add a row to
`_ESTIMATORS`.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from sklearn.base import BaseEstimator

from src.config import ModelType, Task

# (module, classification class, regression class)
_ESTIMATORS: dict[str, tuple[str, str, str]] = {
    "dummy": ("sklearn.dummy", "DummyClassifier", "DummyRegressor"),
    "linear": ("sklearn.linear_model", "LogisticRegression", "Ridge"),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier", "RandomForestRegressor"),
    "hist_gradient_boosting": (
        "sklearn.ensemble",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
    ),
    # XGBoost isn't in requirements.txt yet; see the note there. Its classifier
    # needs integer labels 0..k-1, so map string targets in a cleaner in src/prepare.py.
    "xgboost": ("xgboost", "XGBClassifier", "XGBRegressor"),
}

# Defaults applied unless the config sets them.
_DEFAULT_PARAMS: dict[tuple[str, str], dict[str, Any]] = {
    ("linear", "classification"): {"max_iter": 1000},
}


def build_estimator(
    task: Task, model_type: ModelType, params: dict[str, Any], seed: int
) -> BaseEstimator:
    module_name, classifier, regressor = _ESTIMATORS[model_type]
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ImportError(
            f"model.type '{model_type}' needs the '{module_name}' package, which isn't installed"
        ) from err
    cls = getattr(module, classifier if task == "classification" else regressor)

    kwargs = {**_DEFAULT_PARAMS.get((model_type, task), {}), **params}
    accepted = inspect.signature(cls).parameters
    unknown = sorted(set(kwargs) - set(accepted))
    if unknown:
        raise ValueError(f"{cls.__name__} doesn't accept model.params {unknown}")
    if "random_state" in accepted and "random_state" not in kwargs:
        kwargs["random_state"] = seed
    return cls(**kwargs)
