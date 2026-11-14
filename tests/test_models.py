import inspect

import pytest
from sklearn.base import is_classifier, is_regressor

from src.models import build_estimator


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    "model_type", ["dummy", "linear", "random_forest", "hist_gradient_boosting"]
)
def test_each_model_type_builds_for_each_task(task: str, model_type: str) -> None:
    estimator = build_estimator(task, model_type, {}, seed=7)

    assert (is_classifier if task == "classification" else is_regressor)(estimator)
    if "random_state" in inspect.signature(type(estimator)).parameters:
        assert estimator.random_state == 7


def test_config_params_override_defaults() -> None:
    assert build_estimator("classification", "linear", {"max_iter": 50}, seed=0).max_iter == 50


def test_unknown_params_are_rejected_with_their_names() -> None:
    with pytest.raises(ValueError, match=r"doesn't accept model.params \['n_trees'\]"):
        build_estimator("classification", "random_forest", {"n_trees": 10}, seed=0)
