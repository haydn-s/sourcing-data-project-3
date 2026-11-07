"""Load and validate the pipeline configuration.

A config file can inherit from another one with `extends:` (a path relative to
the file) and override only what changes. Mappings merge key by key, except
`model.params`, which is replaced as a whole so each variant lists exactly the
hyperparameters it uses. Single values can also be overridden from the command
line with `--set key.path=value`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Task = Literal["classification", "regression"]
ModelType = Literal["dummy", "linear", "random_forest", "hist_gradient_boosting", "xgboost"]

METRICS: dict[str, tuple[str, ...]] = {
    "classification": ("accuracy", "f1", "roc_auc"),
    "regression": ("rmse", "mae", "r2"),
}

# Merged paths that a child config replaces instead of merging into.
_REPLACE_ON_MERGE = {("model", "params")}


class _Section(BaseModel):
    # Unknown keys are almost always typos, so reject them.
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    remote_uri: str | None = None
    local_path: Path
    target: str


class FeaturesConfig(_Section):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @model_validator(mode="after")
    def _check_features(self) -> FeaturesConfig:
        if not self.all:
            raise ValueError("list at least one numeric or categorical feature")
        repeated = sorted({name for name in self.all if self.all.count(name) > 1})
        if repeated:
            raise ValueError(f"features listed more than once: {repeated}")
        return self


class SplitConfig(_Section):
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    stratify: bool = True  # classification only

    @model_validator(mode="after")
    def _check_sizes(self) -> SplitConfig:
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must leave rows for training")
        return self


class ModelConfig(_Section):
    type: ModelType
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(_Section):
    primary_metric: str


class TrackingConfig(_Section):
    project: str
    entity: str | None = None  # W&B team; None uses your default entity
    tags: list[str] = Field(default_factory=list)


class OutputConfig(_Section):
    local_dir: Path = Path("models")
    remote_prefix: str | None = None


class PipelineConfig(_Section):
    seed: int = 42
    task: Task
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> PipelineConfig:
        if self.data.target in self.features.all:
            raise ValueError(f"the target '{self.data.target}' is also listed as a feature")
        allowed = METRICS[self.task]
        if self.evaluation.primary_metric not in allowed:
            raise ValueError(
                f"primary_metric '{self.evaluation.primary_metric}' isn't available for "
                f"{self.task}. Choose one of: {', '.join(allowed)}"
            )
        for name, uri in (
            ("data.remote_uri", self.data.remote_uri),
            ("output.remote_prefix", self.output.remote_prefix),
        ):
            if uri is not None and not uri.startswith("s3://"):
                raise ValueError(f"{name} must be an s3:// URI or null, got '{uri}'")
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    """Read a config file, resolve `extends:`, apply `--set` overrides, and validate."""
    raw = _read_with_parents(Path(path), seen=())
    for override in overrides or []:
        _apply_override(raw, override)
    return PipelineConfig.model_validate(raw)


def _read_with_parents(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"config files extend each other in a loop: {chain}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _read_with_parents(path.parent / parent, seen=(*seen, path))
    return _merge(base, data)


def _merge(base: dict[str, Any], override: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        here = (*prefix, key)
        mergeable = isinstance(value, dict) and isinstance(merged.get(key), dict)
        if mergeable and here not in _REPLACE_ON_MERGE:
            merged[key] = _merge(merged[key], value, here)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one `key.path=value` override. The value is parsed as YAML (5, true, [a, b])."""
    key_path, sep, value_text = override.partition("=")
    if not sep or not key_path:
        raise ValueError(f"--set expects key.path=value, got '{override}'")
    *parents, leaf = key_path.split(".")
    node = raw
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {key_path}: '{key}' isn't a mapping")
    node[leaf] = yaml.safe_load(value_text)
