"""ML pipeline: train, evaluate, and publish a model from the processed table.

    python -m ml_pipeline.train --config config.yaml

It downloads the table the data pipeline pushed to `data.remote_uri`, splits
it, fits the preprocessing and model as one scikit-learn Pipeline, logs the run
to W&B, and saves the model where the API (api/) loads it. This package imports
only from `shared`, never from `data_pipeline`.
"""
