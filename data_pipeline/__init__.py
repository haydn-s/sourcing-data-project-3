"""Data pipeline: turn public data sources into the processed table the ML pipeline trains on.

Three stages, each a command (`make data` runs them in order):

1. ingest:  python -m data_pipeline.ingest   fetch each source and save raw snapshots
2. prepare: python -m data_pipeline.prepare  clean, join, and add engineered features
3. push:    python -m data_pipeline.push     upload the processed table to Cloud Storage

The hand-off to the ML pipeline is that table at `data.remote_uri`. This
package imports only from `shared`, never from `ml_pipeline`.
"""
