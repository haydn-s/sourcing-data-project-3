"""Generate a small synthetic dataset so the pipeline runs before we pick real data.

The data imitates customer churn: three numeric and two categorical features, a
few missing values to exercise the imputers, a yes/no target (`churned`) for
classification, and a numeric target (`lifetime_value`) for regression. None of
it describes real people.

    python -m ml_pipeline.demo_data              # writes data/demo/demo.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/demo/demo.csv")


def make_demo_frame(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, n_rows).astype(float)
    region = rng.choice(["north", "south", "east", "west"], n_rows)
    plan = rng.choice(["basic", "standard", "premium"], n_rows, p=[0.5, 0.3, 0.2])
    plan_tier = pd.Series(plan).map({"basic": 0, "standard": 1, "premium": 2}).to_numpy()
    support_calls = rng.poisson(1.5, n_rows)
    monthly_spend = np.clip(20 + 30 * plan_tier + rng.normal(0, 8, n_rows), 5, None).round(2)

    logit = (
        -1.0
        + 0.6 * support_calls
        - 0.03 * (age - 45)
        - 0.4 * (plan_tier == 2)
        + 0.3 * (region == "south")
        + rng.normal(0, 0.5, n_rows)
    )
    churned = np.where(rng.random(n_rows) < 1 / (1 + np.exp(-logit)), "yes", "no")
    months = np.clip(24 - 4 * support_calls + rng.normal(0, 3, n_rows), 1, None)
    lifetime_value = (monthly_spend * months).round(2)

    frame = pd.DataFrame(
        {
            "age": age,
            "region": region,
            "plan": plan,
            "monthly_spend": monthly_spend,
            "support_calls": support_calls,
            "churned": churned,
            "lifetime_value": lifetime_value,
        }
    )
    # Knock out ~5% of two columns so the imputers have work to do.
    frame.loc[rng.random(n_rows) < 0.05, "age"] = np.nan
    frame.loc[rng.random(n_rows) < 0.05, "plan"] = np.nan
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH, help="CSV path to write")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_demo_frame(args.rows, args.seed).to_csv(args.out, index=False)
    print(f"Wrote {args.rows} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
