"""Code both pipelines (and the API) use: the config, Cloud Storage, table files, and CLI plumbing.

Nothing here imports from data_pipeline or ml_pipeline, so either side can change
without touching the other. tests/test_layout.py checks that.
"""
