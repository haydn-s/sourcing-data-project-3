# ML pipeline

Turns the processed table into an evaluated model bundle that the API serves.
Run `make train` (or `python -m ml_pipeline.train --config config.yaml`) after
the data pipeline has uploaded its processed table.

## Run and configure training

- Set `data.remote_uri` to the exact CSV or Parquet object uploaded by the data pipeline. Set `data.local_path`, `data.target`, and `features` to match that table. Downloads use `shared.storage.download()` and `shared.tables.read_table()`; public GCS objects need no login.
- `task` selects classification or regression. `model.type` selects `dummy`, `linear`, `random_forest`, `hist_gradient_boosting`, or `xgboost`; `model.params` are passed to that estimator, with `seed` used as its random state when supported. XGBoost must be installed and pinned in `requirements.txt` before selecting it.
- `split.test_size` and `split.val_size` are fractions of the full table. The model fits on the training split, then the configured `evaluation.primary_metric` is reported for validation and test.
- `tracking` configures the W&B project and team. Set up W&B authentication before running, or set `WANDB_MODE=disabled` to train without recording a W&B run.
- Each run writes `model.joblib` and `metadata.json` under `output.local_dir/<run-id>/`, and updates `output.local_dir/latest/`. If `output.remote_prefix` is set, both artifacts are uploaded under `<prefix>/<run-id>/` and `<prefix>/latest/`; publishing requires Google Cloud write credentials.

## What the API needs from it

[`api/main.py`](../api/main.py) loads two files next to the model artifact.
Point `MODEL_URI` at the **file** `model.joblib`, not its containing folder:

- **`model.joblib`:** an object saved with `joblib.dump`, usually a scikit-learn `Pipeline` that includes its own preprocessing. The API calls `.predict()` on a one-row DataFrame whose columns are the feature names, and `.predict_proba()` too for classification.
- **`metadata.json`:** the API builds its `/predict` request schema from this file. It needs:
  - `model_version`, `created_at`, `task` (`classification` or `regression`), `target`, `model_type`, `primary_metric`
  - `metrics`, e.g. `{"val": {"f1": 0.8}, "test": {"f1": 0.78}}`
  - `features`: a list of `{"name", "kind": "numeric", "min", "max", "example"}` or `{"name", "kind": "categorical", "categories", "example"}`
  - `example`: one valid request, keyed by feature name
  - `classes` (classification only), and `versions` with the `scikit-learn` version, which the API checks against its own

For example, set `output.remote_prefix` to
`gs://climate-impact-pipeline-data/models`, then set the deployed API's
`MODEL_URI` to
`gs://climate-impact-pipeline-data/models/latest/model.joblib`. Uploads need a
Google login with write access to the bucket. See "Set up your .env" in the
[main README](../README.md#set-up-your-env).

## Boundaries

- **Imports:** this package may import from `shared` only, never from `data_pipeline`. `tests/test_layout.py` checks that. Put tests in `tests/ml_pipeline/`.
- **Tracking:** runs log to the configured W&B project and team (`tracking.project` and `tracking.entity`).
- **Tests:** `tests/ml_pipeline/` covers data loading, metric calculation, and the saved bundle. Docker and deployed-API smoke tests can be added separately.
