import numpy as np
import pandas as pd
import pytest

from ml_pipeline.features import build_preprocessor
from shared.config import FeaturesConfig


def test_imputes_missing_values_and_ignores_unseen_categories() -> None:
    features = FeaturesConfig(numeric=["x"], categorical=["color"])
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "color": ["red", "blue", None]})
    preprocessor = build_preprocessor(features).fit(train)

    out = preprocessor.transform(pd.DataFrame({"x": [np.nan], "color": ["green"]}))

    assert out.shape == (1, 3)  # x, plus one column per training color
    assert not np.isnan(out).any()
    assert out[0, 1:].tolist() == [0.0, 0.0]  # an unseen color encodes as all zeros


@pytest.mark.parametrize(
    ("features", "width"),
    [
        (FeaturesConfig(numeric=["x"]), 1),
        (FeaturesConfig(categorical=["color"]), 2),
    ],
)
def test_works_with_only_one_kind_of_feature(features: FeaturesConfig, width: int) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "color": ["red", "blue"]})
    assert build_preprocessor(features).fit_transform(frame).shape == (2, width)


def test_drops_columns_not_in_the_config() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "unused": ["a", "b"]})
    assert build_preprocessor(FeaturesConfig(numeric=["x"])).fit_transform(frame).shape == (2, 1)
