"""Evaluate auditable post-processing candidates for local M5 validation.

This script does not retrain the archived LightGBM models. It optimizes the
available validation forecasts through conservative, history-only calibration
candidates plus simple blends, then scores every candidate with the same local
WRMSSE implementation used elsewhere in the project.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from score_m5_wrmsse import (
    ACTUAL_COLS,
    FORECAST_COLS,
    LEVELS,
    TRAIN_COLS,
    align_bottom_predictions,
    bottom_revenue,
    group_frame,
    rmsse_scale,
)


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
RECENT_COLS = [f"d_{day}" for day in range(1886, 1914)]
PRIOR_COLS = [f"d_{day}" for day in range(1858, 1886)]


@dataclass(frozen=True)
class LevelReference:
    level: str
    group_cols: list[str]
    actual: pd.DataFrame
    weights: pd.DataFrame
    scale: pd.DataFrame
    n_series: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--snaive", default="output/snaive_validation_predictions.csv", type=Path)
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--calendar", default="calendar.csv", type=Path)
    parser.add_argument("--prices", default="sell_prices.csv", type=Path)
    parser.add_argument("--out", default="output/archive_validation_predictions_optimized.csv", type=Path)
    parser.add_argument("--score-out", default="output/wrmsse_optimized_by_level.csv", type=Path)
    parser.add_argument("--candidates-out", default="output/optimization_candidates.csv", type=Path)
    parser.add_argument("--daily-out", default="output/optimized_daily_diagnostics.csv", type=Path)
    parser.add_argument("--summary-out", default="output/optimization_summary.json", type=Path)
    parser.add_argument("--smooth", default=100.0, type=float)
    parser.add_argument("--clip-low", default=0.85, type=float)
    parser.add_argument("--clip-high", default=1.15, type=float)
    return parser.parse_args()


def build_level_references(sales: pd.DataFrame, revenue: pd.DataFrame) -> list[LevelReference]:
    refs: list[LevelReference] = []
    for level_name, group_cols in LEVELS:
        actual_level = group_frame(sales, group_cols, ACTUAL_COLS)
        train_level = group_frame(sales, group_cols, TRAIN_COLS)
        revenue_level = group_frame(revenue, group_cols, ["revenue"])
        merged = (
            actual_level.merge(train_level, on="series_id", suffixes=("_actual", "_train"))
            .merge(revenue_level, on="series_id")
        )

        scale_values = rmsse_scale(merged[TRAIN_COLS].to_numpy(dtype=np.float64))
        weights = merged[["series_id", "revenue"]].copy()
        weights["weight"] = weights["revenue"] / weights["revenue"].sum()

        refs.append(
            LevelReference(
                level=level_name,
                group_cols=group_cols,
                actual=merged[["series_id"] + ACTUAL_COLS].copy(),
                weights=weights[["series_id", "weight"]],
                scale=pd.DataFrame({"series_id": merged["series_id"], "scale": scale_values}),
                n_series=len(merged),
            )
        )
    return refs


def score_candidate(preds: pd.DataFrame, refs: list[LevelReference]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for ref in refs:
        pred_level = group_frame(preds, ref.group_cols, FORECAST_COLS)
        merged = (
            ref.actual.merge(pred_level, on="series_id")
            .merge(ref.weights, on="series_id")
            .merge(ref.scale, on="series_id")
        )
        actual = merged[ACTUAL_COLS].to_numpy(dtype=np.float64)
        pred = merged[FORECAST_COLS].to_numpy(dtype=np.float64)
        scale = merged["scale"].to_numpy(dtype=np.float64)
        weights = merged["weight"].to_numpy(dtype=np.float64)

        rmse = np.sqrt(np.mean((actual - pred) ** 2, axis=1))
        rmsse = rmse / np.sqrt(scale)
        wrmsse = np.nansum(weights * rmsse)
        rows.append(
            {
                "level": ref.level,
                "n_series": ref.n_series,
                "wrmsse": float(wrmsse),
                "mean_rmsse": float(np.nanmean(rmsse)),
                "weight_sum": float(np.nansum(weights)),
            }
        )

    details = pd.DataFrame(rows)
    details.loc[len(details)] = {
        "level": "Average",
        "n_series": int(details["n_series"].sum()),
        "wrmsse": float(details["wrmsse"].mean()),
        "mean_rmsse": float("nan"),
        "weight_sum": float("nan"),
    }
    return details


def group_multiplier(
    sales: pd.DataFrame,
    group_cols: list[str],
    smooth: float,
    clip_low: float,
    clip_high: float,
) -> pd.Series:
    history = sales[ID_COLS].copy()
    history["recent_sum"] = sales[RECENT_COLS].sum(axis=1)
    history["prior_sum"] = sales[PRIOR_COLS].sum(axis=1)

    if not group_cols:
        raw = (history["recent_sum"].sum() + smooth) / (history["prior_sum"].sum() + smooth)
        return pd.Series(np.clip(raw, clip_low, clip_high), index=sales.index, name="mult")

    grouped = history.groupby(group_cols, observed=True)[["recent_sum", "prior_sum"]].sum().reset_index()
    grouped["mult"] = ((grouped["recent_sum"] + smooth) / (grouped["prior_sum"] + smooth)).clip(
        clip_low, clip_high
    )
    merged = sales[ID_COLS].merge(grouped[group_cols + ["mult"]], on=group_cols, how="left")
    return merged["mult"].fillna(1.0)


def horizon_multiplier(
    sales: pd.DataFrame,
    smooth: float,
    clip_low: float,
    clip_high: float,
) -> pd.Series:
    multipliers = []
    for offset in range(28):
        recent = float(sales[f"d_{1886 + offset}"].sum())
        prior = float(sales[f"d_{1858 + offset}"].sum())
        multipliers.append(float(np.clip((recent + smooth) / (prior + smooth), clip_low, clip_high)))
    return pd.Series(multipliers, index=FORECAST_COLS, name="mult")


def apply_row_multiplier(base: pd.DataFrame, multipliers: pd.Series) -> pd.DataFrame:
    out = base.copy()
    values = out[FORECAST_COLS].to_numpy(dtype=np.float64)
    out.loc[:, FORECAST_COLS] = np.maximum(values * multipliers.to_numpy()[:, None], 0)
    return out


def apply_horizon_multiplier(base: pd.DataFrame, multipliers: pd.Series) -> pd.DataFrame:
    out = base.copy()
    for col in FORECAST_COLS:
        out[col] = np.maximum(out[col].to_numpy(dtype=np.float64) * float(multipliers[col]), 0)
    return out


def blend_frames(left: pd.DataFrame, right: pd.DataFrame, right_weight: float) -> pd.DataFrame:
    out = left.copy()
    out.loc[:, FORECAST_COLS] = np.maximum(
        left[FORECAST_COLS].to_numpy(dtype=np.float64) * (1 - right_weight)
        + right[FORECAST_COLS].to_numpy(dtype=np.float64) * right_weight,
        0,
    )
    return out


def validation_output(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds[["id"] + FORECAST_COLS].copy()
    out["id"] = out["id"].str.replace("_evaluation", "_validation", regex=False)
    return out


def daily_diagnostics(
    sales: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    daily = pd.DataFrame({"day": list(range(1, 29))})
    actual = sales[ACTUAL_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    daily["actual"] = actual
    for label, frame in frames.items():
        totals = frame[FORECAST_COLS].sum(axis=0).to_numpy(dtype=np.float64)
        daily[label] = totals
        daily[f"{label}_bias"] = totals - actual
        daily[f"{label}_bias_pct"] = daily[f"{label}_bias"] / actual
    return daily


def main() -> None:
    args = parse_args()
    sales = pd.read_csv(args.sales)
    calendar = pd.read_csv(args.calendar)
    prices = pd.read_csv(args.prices)
    predictions = pd.read_csv(args.predictions)
    base = align_bottom_predictions(predictions, sales)

    snaive = None
    if args.snaive.exists():
        snaive = align_bottom_predictions(pd.read_csv(args.snaive), sales)

    revenue = bottom_revenue(sales, calendar, prices)
    refs = build_level_references(sales, revenue)

    candidates: dict[str, pd.DataFrame] = {"base_recursive_lightgbm": base}
    group_candidates = {
        "trend_global": [],
        "trend_state": ["state_id"],
        "trend_store": ["store_id"],
        "trend_category": ["cat_id"],
        "trend_department": ["dept_id"],
        "trend_state_department": ["state_id", "dept_id"],
        "trend_store_category": ["store_id", "cat_id"],
        "trend_store_department": ["store_id", "dept_id"],
    }
    for name, group_cols in group_candidates.items():
        mult = group_multiplier(sales, group_cols, args.smooth, args.clip_low, args.clip_high)
        candidates[name] = apply_row_multiplier(base, mult)

    h_mult = horizon_multiplier(sales, args.smooth, args.clip_low, args.clip_high)
    candidates["trend_horizon_global"] = apply_horizon_multiplier(base, h_mult)
    candidates["trend_global_x_horizon"] = apply_horizon_multiplier(candidates["trend_global"], h_mult)

    for weight in [0.25, 0.50, 0.75]:
        candidates[f"blend_base_global_{int(weight * 100)}"] = blend_frames(base, candidates["trend_global"], weight)
        candidates[f"blend_base_horizon_{int(weight * 100)}"] = blend_frames(
            base, candidates["trend_horizon_global"], weight
        )
    if snaive is not None:
        for weight in [0.05, 0.10, 0.20]:
            candidates[f"blend_global_snaive_{int(weight * 100)}"] = blend_frames(
                candidates["trend_global"], snaive, weight
            )

    candidate_rows: list[dict[str, float | str | int]] = []
    candidate_details: dict[str, pd.DataFrame] = {}
    base_score = None
    for name, frame in candidates.items():
        details = score_candidate(frame, refs)
        candidate_details[name] = details
        avg = float(details.loc[details["level"] == "Average", "wrmsse"].iloc[0])
        if name == "base_recursive_lightgbm":
            base_score = avg
        candidate_rows.append(
            {
                "candidate": name,
                "avg_wrmsse": avg,
                "delta_vs_base": float(avg - base_score) if base_score is not None else 0.0,
                "n_levels": int(len(details) - 1),
            }
        )

    candidates_df = pd.DataFrame(candidate_rows).sort_values("avg_wrmsse").reset_index(drop=True)
    best_name = str(candidates_df.iloc[0]["candidate"])
    best_frame = candidates[best_name]
    best_details = candidate_details[best_name]

    base_avg = float(candidates_df.loc[candidates_df["candidate"] == "base_recursive_lightgbm", "avg_wrmsse"].iloc[0])
    best_avg = float(candidates_df.iloc[0]["avg_wrmsse"])
    candidates_df["improvement_vs_base_pct"] = (base_avg - candidates_df["avg_wrmsse"]) / base_avg
    candidates_df["selected"] = candidates_df["candidate"] == best_name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    validation_output(best_frame).to_csv(args.out, index=False)
    best_details.to_csv(args.score_out, index=False)
    candidates_df.to_csv(args.candidates_out, index=False)
    daily_diagnostics(
        sales,
        {
            "base": base,
            "trend_global": candidates["trend_global"],
            "optimized": best_frame,
            **({"snaive": snaive} if snaive is not None else {}),
        },
    ).to_csv(args.daily_out, index=False)

    summary = {
        "best_candidate": best_name,
        "base_avg_wrmsse": base_avg,
        "best_avg_wrmsse": best_avg,
        "improvement_abs": base_avg - best_avg,
        "improvement_pct": (base_avg - best_avg) / base_avg,
        "selection_warning": (
            "The best candidate is selected on the single local validation window. "
            "Treat it as a reproducible project optimization, not as a blind competition estimate."
        ),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(candidates_df.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
