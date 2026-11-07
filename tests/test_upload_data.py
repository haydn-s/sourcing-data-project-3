from pathlib import Path

import pandas as pd

from scripts.upload_data import main
from src.demo_data import make_demo_frame
from tests.conftest import REPO_CONFIG


def test_cleans_raw_data_and_writes_the_processed_file(tmp_path: Path) -> None:
    raw = make_demo_frame(n_rows=50)
    raw_path = tmp_path / "raw.csv"
    pd.concat([raw, raw.head(5)]).to_csv(raw_path, index=False)  # 5 duplicate rows
    processed_path = tmp_path / "processed" / "data.parquet"
    config = tmp_path / "config.yaml"
    config.write_text(f"extends: {REPO_CONFIG}\ndata:\n  local_path: {processed_path}\n")

    main(["--config", str(config), "--raw", str(raw_path)])

    processed = pd.read_parquet(processed_path)
    assert len(processed) == 50
    assert not processed.duplicated().any()
