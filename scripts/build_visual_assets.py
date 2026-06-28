"""Build compact CSV/JSON assets for the M5 project dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from score_m5_wrmsse import ACTUAL_COLS, FORECAST_COLS, align_bottom_predictions


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
ALL_DAY_COLS = [f"d_{day}" for day in range(1, 1942)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--calendar", default="calendar.csv", type=Path)
    parser.add_argument("--prices", default="sell_prices.csv", type=Path)
    parser.add_argument("--base-pred", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--optimized-pred", default="output/archive_validation_predictions_optimized.csv", type=Path)
    parser.add_argument("--candidates", default="output/optimization_candidates.csv", type=Path)
    parser.add_argument("--out-dir", default="output/visual", type=Path)
    return parser.parse_args()


def normalize_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()
    cal["day"] = cal["d"].str.extract(r"(\d+)").astype(int)
    cal["event_label"] = cal["event_name_1"].fillna("")
    cal.loc[cal["event_name_2"].notna(), "event_label"] = (
        cal.loc[cal["event_name_2"].notna(), "event_label"]
        + " / "
        + cal.loc[cal["event_name_2"].notna(), "event_name_2"].astype(str)
    )
    cal["snap_total"] = cal[["snap_CA", "snap_TX", "snap_WI"]].sum(axis=1)
    return cal


def build_daily_sales(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    total = sales[ALL_DAY_COLS].sum(axis=0).reset_index()
    total.columns = ["d", "sales"]
    daily = total.merge(
        calendar[
            [
                "d",
                "day",
                "date",
                "weekday",
                "wday",
                "month",
                "year",
                "event_label",
                "event_type_1",
                "snap_total",
            ]
        ],
        on="d",
        how="left",
    )
    daily["roll_7"] = daily["sales"].rolling(7, min_periods=1).mean()
    daily["roll_28"] = daily["sales"].rolling(28, min_periods=1).mean()
    return daily


def build_sales_mix(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_cols = ACTUAL_COLS
    store = sales.groupby(["state_id", "store_id"], observed=True)[value_cols].sum().sum(axis=1).reset_index()
    store.columns = ["state_id", "store_id", "validation_sales"]
    store["share"] = store["validation_sales"] / store["validation_sales"].sum()

    category = sales.groupby(["cat_id", "dept_id"], observed=True)[value_cols].sum().sum(axis=1).reset_index()
    category.columns = ["cat_id", "dept_id", "validation_sales"]
    category["share"] = category["validation_sales"] / category["validation_sales"].sum()

    state_category = (
        sales.groupby(["state_id", "cat_id"], observed=True)[value_cols].sum().sum(axis=1).reset_index()
    )
    state_category.columns = ["state_id", "cat_id", "validation_sales"]
    state_category["share"] = state_category["validation_sales"] / state_category["validation_sales"].sum()
    return store, category, state_category


def build_event_summary(daily: pd.DataFrame) -> pd.DataFrame:
    event = daily.copy()
    event["event_group"] = event["event_type_1"].fillna("No event")
    summary = (
        event.groupby("event_group", observed=True)
        .agg(days=("sales", "size"), avg_sales=("sales", "mean"), median_sales=("sales", "median"))
        .reset_index()
        .sort_values("avg_sales", ascending=False)
    )
    return summary


def build_price_summary(prices: pd.DataFrame) -> pd.DataFrame:
    summary = (
        prices.groupby("store_id", observed=True)
        .agg(
            rows=("sell_price", "size"),
            items=("item_id", "nunique"),
            avg_price=("sell_price", "mean"),
            median_price=("sell_price", "median"),
            min_price=("sell_price", "min"),
            max_price=("sell_price", "max"),
        )
        .reset_index()
    )
    return summary


def build_forecast_bias(
    sales: pd.DataFrame,
    base_pred: pd.DataFrame,
    optimized_pred: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    base = align_bottom_predictions(base_pred, sales)
    optimized = align_bottom_predictions(optimized_pred, sales)

    actual_group = sales[ID_COLS + ACTUAL_COLS].copy()
    actual_group["actual"] = actual_group[ACTUAL_COLS].sum(axis=1)
    base_group = base[ID_COLS + FORECAST_COLS].copy()
    base_group["base"] = base_group[FORECAST_COLS].sum(axis=1)
    opt_group = optimized[ID_COLS + FORECAST_COLS].copy()
    opt_group["optimized"] = opt_group[FORECAST_COLS].sum(axis=1)

    merged = actual_group[ID_COLS + ["actual"]].merge(base_group[ID_COLS + ["base"]], on=ID_COLS).merge(
        opt_group[ID_COLS + ["optimized"]], on=ID_COLS
    )
    grouped = merged.groupby(group_cols, observed=True)[["actual", "base", "optimized"]].sum().reset_index()
    grouped["base_bias_pct"] = (grouped["base"] - grouped["actual"]) / grouped["actual"]
    grouped["optimized_bias_pct"] = (grouped["optimized"] - grouped["actual"]) / grouped["actual"]
    grouped["bias_improvement_abs"] = grouped["base_bias_pct"].abs() - grouped["optimized_bias_pct"].abs()
    return grouped.sort_values("optimized_bias_pct")


def build_example_series(
    sales: pd.DataFrame,
    base_pred: pd.DataFrame,
    optimized_pred: pd.DataFrame,
) -> pd.DataFrame:
    base = align_bottom_predictions(base_pred, sales)
    optimized = align_bottom_predictions(optimized_pred, sales)
    top_ids = (
        sales.assign(validation_total=sales[ACTUAL_COLS].sum(axis=1))
        .sort_values("validation_total", ascending=False)
        .head(5)[ID_COLS]
    )
    rows = []
    for _, item in top_ids.iterrows():
        sid = item["id"]
        sales_row = sales.loc[sales["id"] == sid].iloc[0]
        base_row = base.loc[base["id"] == sid].iloc[0]
        opt_row = optimized.loc[optimized["id"] == sid].iloc[0]
        for idx in range(28):
            rows.append(
                {
                    **{col: item[col] for col in ID_COLS},
                    "day": idx + 1,
                    "actual": float(sales_row[ACTUAL_COLS[idx]]),
                    "base": float(base_row[FORECAST_COLS[idx]]),
                    "optimized": float(opt_row[FORECAST_COLS[idx]]),
                }
            )
    return pd.DataFrame(rows)


def build_feature_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("importance*.png")):
        stem = path.stem
        kind = "split" if stem.startswith("importance_split") else "gain"
        store = stem.replace("importance_split_", "").replace("importance_", "")
        rows.append({"file": path.name, "store_id": store, "kind": kind})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(args.sales)
    calendar = normalize_calendar(pd.read_csv(args.calendar))
    prices = pd.read_csv(args.prices)
    base_pred = pd.read_csv(args.base_pred)
    optimized_pred = pd.read_csv(args.optimized_pred)

    daily = build_daily_sales(sales, calendar)
    store, category, state_category = build_sales_mix(sales)
    event_summary = build_event_summary(daily)
    price_summary = build_price_summary(prices)
    store_bias = build_forecast_bias(sales, base_pred, optimized_pred, ["state_id", "store_id"])
    category_bias = build_forecast_bias(sales, base_pred, optimized_pred, ["cat_id"])
    examples = build_example_series(sales, base_pred, optimized_pred)
    inventory = build_feature_inventory(Path("."))

    daily.to_csv(args.out_dir / "eda_daily_sales.csv", index=False)
    store.to_csv(args.out_dir / "eda_store_sales.csv", index=False)
    category.to_csv(args.out_dir / "eda_category_sales.csv", index=False)
    state_category.to_csv(args.out_dir / "eda_state_category_sales.csv", index=False)
    event_summary.to_csv(args.out_dir / "eda_event_summary.csv", index=False)
    price_summary.to_csv(args.out_dir / "eda_price_summary.csv", index=False)
    store_bias.to_csv(args.out_dir / "forecast_store_bias.csv", index=False)
    category_bias.to_csv(args.out_dir / "forecast_category_bias.csv", index=False)
    examples.to_csv(args.out_dir / "forecast_example_series.csv", index=False)
    inventory.to_csv(args.out_dir / "feature_importance_inventory.csv", index=False)

    summary = {
        "bottom_series": int(len(sales)),
        "states": int(sales["state_id"].nunique()),
        "stores": int(sales["store_id"].nunique()),
        "categories": int(sales["cat_id"].nunique()),
        "departments": int(sales["dept_id"].nunique()),
        "items": int(sales["item_id"].nunique()),
        "days": int(len(ALL_DAY_COLS)),
        "validation_actual_sales": float(sales[ACTUAL_COLS].sum().sum()),
        "feature_importance_images": int(len(inventory)),
        "optimization_candidates": int(len(pd.read_csv(args.candidates))) if args.candidates.exists() else None,
    }
    (args.out_dir / "visual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
