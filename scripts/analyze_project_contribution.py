"""Summarize the M5 project contribution and forecasting diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FORECAST_COLS = [f"F{day}" for day in range(1, 29)]
ACTUAL_COLS = [f"d_{day}" for day in range(1914, 1942)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--snaive-score", default="output/wrmsse_snaive_by_level.csv", type=Path)
    parser.add_argument("--model-score", default="output/wrmsse_by_level.csv", type=Path)
    parser.add_argument("--trend-score", default="output/wrmsse_trend_global_by_level.csv", type=Path)
    parser.add_argument("--optimized-score", default="output/wrmsse_optimized_by_level.csv", type=Path)
    parser.add_argument("--snaive-pred", default="output/snaive_validation_predictions.csv", type=Path)
    parser.add_argument("--model-pred", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--trend-pred", default="output/archive_validation_predictions_trend_global.csv", type=Path)
    parser.add_argument("--optimized-pred", default="output/archive_validation_predictions_optimized.csv", type=Path)
    parser.add_argument("--level-out", default="output/level_contribution.csv", type=Path)
    parser.add_argument("--daily-out", default="output/daily_totals.csv", type=Path)
    parser.add_argument("--summary-out", default="output/project_summary.json", type=Path)
    return parser.parse_args()


def read_score(path: Path, suffix: str) -> pd.DataFrame:
    score = pd.read_csv(path)[["level", "n_series", "wrmsse"]]
    return score.rename(columns={"wrmsse": f"wrmsse_{suffix}"})


def prediction_daily_totals(path: Path, label: str) -> pd.DataFrame:
    preds = pd.read_csv(path)
    totals = preds[FORECAST_COLS].sum(axis=0).reset_index()
    totals.columns = ["horizon", label]
    totals["day"] = totals["horizon"].str.extract(r"(\d+)").astype(int)
    return totals.drop(columns=["horizon"])


def main() -> None:
    args = parse_args()
    snaive = read_score(args.snaive_score, "snaive")
    model = read_score(args.model_score, "model").drop(columns=["n_series"])
    trend = read_score(args.trend_score, "trend").drop(columns=["n_series"])
    optimized_score_path = args.optimized_score if args.optimized_score.exists() else args.trend_score
    optimized = read_score(optimized_score_path, "optimized").drop(columns=["n_series"])

    levels = snaive.merge(model, on="level").merge(trend, on="level").merge(optimized, on="level")
    levels["model_gain_abs"] = levels["wrmsse_snaive"] - levels["wrmsse_model"]
    levels["model_gain_pct"] = levels["model_gain_abs"] / levels["wrmsse_snaive"]
    levels["trend_gain_abs"] = levels["wrmsse_model"] - levels["wrmsse_trend"]
    levels["trend_gain_pct"] = levels["trend_gain_abs"] / levels["wrmsse_model"]
    levels["optimized_gain_abs"] = levels["wrmsse_model"] - levels["wrmsse_optimized"]
    levels["optimized_gain_pct"] = levels["optimized_gain_abs"] / levels["wrmsse_model"]
    levels["total_gain_pct"] = (levels["wrmsse_snaive"] - levels["wrmsse_optimized"]) / levels["wrmsse_snaive"]

    sales = pd.read_csv(args.sales)
    actual = sales[ACTUAL_COLS].sum(axis=0).reset_index()
    actual.columns = ["actual_day", "actual"]
    actual["day"] = range(1, 29)
    daily = actual[["day", "actual"]]
    for path, label in [
        (args.snaive_pred, "snaive"),
        (args.model_pred, "model"),
        (args.trend_pred, "trend"),
        (args.optimized_pred if args.optimized_pred.exists() else args.trend_pred, "optimized"),
    ]:
        daily = daily.merge(prediction_daily_totals(path, label), on="day")
    for label in ["snaive", "model", "trend", "optimized"]:
        daily[f"{label}_bias"] = daily[label] - daily["actual"]
        daily[f"{label}_bias_pct"] = daily[f"{label}_bias"] / daily["actual"]

    avg = levels[levels["level"] == "Average"].iloc[0]
    summary = {
        "snaive_avg_wrmsse": float(avg["wrmsse_snaive"]),
        "model_avg_wrmsse": float(avg["wrmsse_model"]),
        "trend_avg_wrmsse": float(avg["wrmsse_trend"]),
        "optimized_avg_wrmsse": float(avg["wrmsse_optimized"]),
        "model_gain_abs": float(avg["model_gain_abs"]),
        "model_gain_pct": float(avg["model_gain_pct"]),
        "trend_gain_abs": float(avg["trend_gain_abs"]),
        "trend_gain_pct": float(avg["trend_gain_pct"]),
        "optimized_gain_abs": float(avg["optimized_gain_abs"]),
        "optimized_gain_pct": float(avg["optimized_gain_pct"]),
        "total_gain_pct": float(avg["total_gain_pct"]),
        "actual_total_sales": float(daily["actual"].sum()),
        "model_total_sales": float(daily["model"].sum()),
        "trend_total_sales": float(daily["trend"].sum()),
        "optimized_total_sales": float(daily["optimized"].sum()),
        "model_total_bias_pct": float((daily["model"].sum() - daily["actual"].sum()) / daily["actual"].sum()),
        "trend_total_bias_pct": float((daily["trend"].sum() - daily["actual"].sum()) / daily["actual"].sum()),
        "optimized_total_bias_pct": float(
            (daily["optimized"].sum() - daily["actual"].sum()) / daily["actual"].sum()
        ),
        "official_top1_avg_wrmsse": 0.5204381930360308,
        "official_top2_avg_wrmsse": 0.5281645743152866,
        "official_snaive_avg_wrmsse": 0.8470173528438624,
        "note": "Local validation d_1914..d_1941; official scores are not the same evaluation slice.",
    }

    args.level_out.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(args.level_out, index=False)
    daily.to_csv(args.daily_out, index=False)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(levels.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
