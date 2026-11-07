"""Model-time preprocessing, driven by the feature lists in the config.

These transforms are part of the saved model pipeline, so the API applies
exactly what training did. Dataset-specific cleaning (dropping columns, fixing
types) happens earlier, in scripts/upload_data.py.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FeaturesConfig


def build_preprocessor(features: FeaturesConfig) -> ColumnTransformer:
    transformers = []
    if features.numeric:
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("num", numeric, features.numeric))
    if features.categorical:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                # Categories never seen in training encode as all zeros instead of erroring.
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical, features.categorical))
    return ColumnTransformer(transformers, remainder="drop")
