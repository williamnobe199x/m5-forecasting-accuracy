"""Apply a pre-validation trend multiplier to M5 validation forecasts.

The multiplier uses only history available up to ``d_1913``:

    sum(d_1886..d_1913) / sum(d_1858..d_1885)

By default it applies one conservative global multiplier. This is intentionally
simple and auditable; broader group-specific multipliers should be selected by
rolling CV before being treated as a production improvement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_COLS = [f"F{day}" for day in range(1, 29)]
RECENT_COLS = [f"d_{day}" for day in range(1886, 1914)]
PRIOR_COLS = [f"d_{day}" for day in range(1858, 1886)]
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--out", default="output/archive_validation_predictions_trend_global.csv", type=Path)
    parser.add_argument(
        "--group-cols",
        default="",
        help="Optional comma-separated group columns, e.g. store_id,dept_id. Empty means global.",
    )
    parser.add_argument("--smooth", default=100.0, type=float)
    parser.add_argument("--clip-low", default=0.85, type=float)
    parser.add_argument("--clip-high", default=1.15, type=float)
    return parser.parse_args()


def build_multiplier(
    sales: pd.DataFrame,
    group_cols: list[str],
    smooth: float,
    clip_low: float,
    clip_high: float,
) -> pd.DataFrame:
    history = sales[ID_COLS].copy()
    history["recent_sum"] = sales[RECENT_COLS].sum(axis=1)
    history["prior_sum"] = sales[PRIOR_COLS].sum(axis=1)

    if not group_cols:
        raw = (history["recent_sum"].sum() + smooth) / (history["prior_sum"].sum() + smooth)
        mult = float(np.clip(raw, clip_low, clip_high))
        return sales[["id"]].assign(mult=mult)

    grouped = history.groupby(group_cols, observed=True)[["recent_sum", "prior_sum"]].sum().reset_index()
    grouped["mult"] = ((grouped["recent_sum"] + smooth) / (grouped["prior_sum"] + smooth)).clip(
        clip_low, clip_high
    )
    return sales[ID_COLS].merge(grouped[group_cols + ["mult"]], on=group_cols, how="left")[["id", "mult"]]


def main() -> None:
    args = parse_args()
    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]

    sales = pd.read_csv(args.sales)
    predictions = pd.read_csv(args.predictions)
    predictions["id"] = predictions["id"].str.replace("_validation", "_evaluation", regex=False)

    multipliers = build_multiplier(sales, group_cols, args.smooth, args.clip_low, args.clip_high)
    adjusted = multipliers.merge(predictions, on="id", how="left")
    if adjusted[FORECAST_COLS].isna().any().any():
        raise ValueError("Some forecasts could not be aligned to sales IDs.")

    output = adjusted[["id"]].copy()
    output["id"] = output["id"].str.replace("_evaluation", "_validation", regex=False)
    for col in FORECAST_COLS:
        output[col] = adjusted[col] * adjusted["mult"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({output.shape})")
    print(
        "multiplier summary:",
        adjusted["mult"].describe()[["mean", "std", "min", "max"]].to_dict(),
    )


if __name__ == "__main__":
    main()
