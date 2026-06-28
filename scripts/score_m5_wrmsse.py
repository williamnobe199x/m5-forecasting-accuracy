"""Compute a local M5-style WRMSSE score for validation predictions.

The expected forecast file has columns ``id,F1,...,F28``. IDs may use either
``_validation`` or ``_evaluation`` suffixes; they are aligned to
``sales_train_evaluation.csv`` by item/store identity.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


TRAIN_END = 1913
HORIZON = 28
TRAIN_COLS = [f"d_{day}" for day in range(1, TRAIN_END + 1)]
ACTUAL_COLS = [f"d_{day}" for day in range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)]
WEIGHT_COLS = [f"d_{day}" for day in range(TRAIN_END - HORIZON + 1, TRAIN_END + 1)]
FORECAST_COLS = [f"F{day}" for day in range(1, HORIZON + 1)]

LEVELS: list[tuple[str, list[str]]] = [
    ("L1_Total", []),
    ("L2_State", ["state_id"]),
    ("L3_Store", ["store_id"]),
    ("L4_Category", ["cat_id"]),
    ("L5_Department", ["dept_id"]),
    ("L6_State_Category", ["state_id", "cat_id"]),
    ("L7_State_Department", ["state_id", "dept_id"]),
    ("L8_Store_Category", ["store_id", "cat_id"]),
    ("L9_Store_Department", ["store_id", "dept_id"]),
    ("L10_Item", ["item_id"]),
    ("L11_Item_State", ["item_id", "state_id"]),
    ("L12_Item_Store", ["item_id", "store_id"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--calendar", default="calendar.csv", type=Path)
    parser.add_argument("--prices", default="sell_prices.csv", type=Path)
    parser.add_argument("--details-out", default="output/wrmsse_by_level.csv", type=Path)
    return parser.parse_args()


def normalize_prediction_id(series: pd.Series) -> pd.Series:
    return series.str.replace("_validation", "_evaluation", regex=False)


def group_frame(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        values = frame[value_cols].sum(axis=0).to_frame().T
        values.insert(0, "series_id", "Total")
        return values

    grouped = frame.groupby(group_cols, observed=True)[value_cols].sum().reset_index()
    grouped.insert(0, "series_id", grouped[group_cols].astype(str).agg("_".join, axis=1))
    return grouped.drop(columns=group_cols)


def rmsse_scale(train_values: np.ndarray) -> np.ndarray:
    scales = np.empty(train_values.shape[0], dtype=np.float64)
    for idx, row in enumerate(train_values):
        nonzero = np.flatnonzero(row)
        if len(nonzero) == 0 or nonzero[0] >= len(row) - 1:
            scales[idx] = np.nan
            continue
        active = row[nonzero[0] :]
        diff = np.diff(active)
        scales[idx] = np.mean(diff * diff)
    return scales


def bottom_revenue(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    melted = sales[id_cols + WEIGHT_COLS].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    day_to_week = calendar[["d", "wm_yr_wk"]]
    melted = melted.merge(day_to_week, on="d", how="left")
    melted = melted.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    melted["revenue"] = melted["sales"] * melted["sell_price"].fillna(0)
    return melted.groupby(id_cols, observed=True)["revenue"].sum().reset_index()


def align_bottom_predictions(predictions: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    preds = predictions.copy()
    preds["id"] = normalize_prediction_id(preds["id"])
    expected = set(["id"] + FORECAST_COLS)
    missing = sorted(expected - set(preds.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    aligned = sales[id_cols].merge(preds[["id"] + FORECAST_COLS], on="id", how="left")
    if aligned[FORECAST_COLS].isna().any().any():
        missing_count = int(aligned[FORECAST_COLS].isna().any(axis=1).sum())
        raise ValueError(f"Missing forecasts for {missing_count} bottom-level series")
    return aligned


def score_level(
    level_name: str,
    group_cols: list[str],
    sales: pd.DataFrame,
    preds: pd.DataFrame,
    revenue: pd.DataFrame,
) -> dict[str, float | int | str]:
    actual_level = group_frame(sales, group_cols, ACTUAL_COLS)
    train_level = group_frame(sales, group_cols, TRAIN_COLS)
    pred_level = group_frame(preds, group_cols, FORECAST_COLS)
    revenue_level = group_frame(revenue, group_cols, ["revenue"])

    merged = (
        actual_level.merge(train_level, on="series_id", suffixes=("_actual", "_train"))
        .merge(pred_level, on="series_id")
        .merge(revenue_level, on="series_id")
    )

    actual = merged[ACTUAL_COLS].to_numpy(dtype=np.float64)
    train = merged[TRAIN_COLS].to_numpy(dtype=np.float64)
    pred = merged[FORECAST_COLS].to_numpy(dtype=np.float64)

    scale = rmsse_scale(train)
    rmse = np.sqrt(np.mean((actual - pred) ** 2, axis=1))
    rmsse = rmse / np.sqrt(scale)

    weights = merged["revenue"].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()
    wrmsse = np.nansum(weights * rmsse)

    return {
        "level": level_name,
        "n_series": len(merged),
        "wrmsse": wrmsse,
        "mean_rmsse": float(np.nanmean(rmsse)),
        "weight_sum": float(weights.sum()),
    }


def main() -> None:
    args = parse_args()
    sales = pd.read_csv(args.sales)
    calendar = pd.read_csv(args.calendar)
    prices = pd.read_csv(args.prices)
    predictions = pd.read_csv(args.predictions)

    preds = align_bottom_predictions(predictions, sales)
    revenue = bottom_revenue(sales, calendar, prices)

    details = pd.DataFrame(
        score_level(level_name, group_cols, sales, preds, revenue)
        for level_name, group_cols in LEVELS
    )
    details.loc[len(details)] = {
        "level": "Average",
        "n_series": int(details["n_series"].sum()),
        "wrmsse": float(details["wrmsse"].mean()),
        "mean_rmsse": float("nan"),
        "weight_sum": float("nan"),
    }

    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_out, index=False)
    print(details.to_string(index=False))
    print(f"wrote {args.details_out}")


if __name__ == "__main__":
    main()
