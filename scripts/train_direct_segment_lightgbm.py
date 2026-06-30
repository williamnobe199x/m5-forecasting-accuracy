"""Train a compact direct-horizon LightGBM pilot.

The model predicts four 7-day segment totals for each item-store series, then
allocates each segment back to daily forecasts using the previous 28-day daily
shape. It is intentionally small enough to run locally and auditable enough for
project reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
DAY_COLS = [f"d_{day}" for day in range(1, 1942)]
FORECAST_COLS = [f"F{day}" for day in range(1, 29)]
SEGMENTS = [(1, 7), (8, 14), (15, 21), (22, 28)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", default="sales_train_evaluation.csv", type=Path)
    parser.add_argument("--out", default="output/direct_segment_lightgbm_predictions.csv", type=Path)
    parser.add_argument("--summary-out", default="output/direct_segment_lightgbm_summary.json", type=Path)
    parser.add_argument("--train-start", default=1400, type=int)
    parser.add_argument("--train-end-origin", default=1886, type=int)
    parser.add_argument("--step", default=28, type=int)
    parser.add_argument("--n-estimators", default=140, type=int)
    parser.add_argument("--num-leaves", default=48, type=int)
    parser.add_argument("--learning-rate", default=0.06, type=float)
    parser.add_argument("--threads", default=-1, type=int)
    return parser.parse_args()


def dcols(start: int, length: int = 28) -> list[str]:
    return [f"d_{day}" for day in range(start, start + length)]


def training_origins(start: int, end_origin: int, step: int) -> list[int]:
    origins = []
    origin = end_origin
    while origin >= start:
        origins.append(origin)
        origin -= step
    return sorted(origins)


def add_group_sum(
    sales: pd.DataFrame,
    features: pd.DataFrame,
    origin: int,
    group_cols: list[str],
    window_start: int,
    window_days: int,
    feature_name: str,
) -> None:
    cols = dcols(origin - window_start, window_days)
    frame = sales[group_cols].copy()
    frame[feature_name] = sales[cols].sum(axis=1)
    grouped = frame.groupby(group_cols, observed=True)[feature_name].sum().reset_index()
    features[feature_name] = sales[group_cols].merge(grouped, on=group_cols, how="left")[feature_name].to_numpy()


def base_feature_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(index=sales.index)
    categorical_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for col in categorical_cols:
        features[col] = sales[col].astype("category").cat.codes.astype("int16")
    return features, categorical_cols


def build_features(sales: pd.DataFrame, origin: int) -> pd.DataFrame:
    features, _ = base_feature_frame(sales)
    features["origin_day"] = np.int16(origin)
    for lag in [1, 7, 14, 28, 56]:
        features[f"lag_{lag}"] = sales[f"d_{origin - lag}"].to_numpy(dtype=np.float32)
    for window in [7, 14, 28, 56]:
        features[f"roll_{window}_sum"] = sales[dcols(origin - window, window)].sum(axis=1).to_numpy(dtype=np.float32)
        features[f"roll_{window}_mean"] = features[f"roll_{window}_sum"] / np.float32(window)
    features["zero_rate_28"] = (sales[dcols(origin - 28, 28)] == 0).mean(axis=1).to_numpy(dtype=np.float32)
    features["trend_28_vs_prior"] = (
        (sales[dcols(origin - 28, 28)].sum(axis=1) + 1)
        / (sales[dcols(origin - 56, 28)].sum(axis=1) + 1)
    ).to_numpy(dtype=np.float32)

    add_group_sum(sales, features, origin, ["item_id"], 28, 28, "item_all_store_roll_28_sum")
    add_group_sum(sales, features, origin, ["dept_id", "store_id"], 28, 28, "dept_store_roll_28_sum")
    add_group_sum(sales, features, origin, ["cat_id", "store_id"], 28, 28, "cat_store_roll_28_sum")
    add_group_sum(sales, features, origin, ["store_id"], 28, 28, "store_roll_28_sum")
    add_group_sum(sales, features, origin, ["state_id", "cat_id"], 28, 28, "state_cat_roll_28_sum")
    add_group_sum(sales, features, origin, ["item_id"], 56, 28, "item_all_store_prior_28_sum")
    features["item_all_store_trend"] = (
        (features["item_all_store_roll_28_sum"] + 1) / (features["item_all_store_prior_28_sum"] + 1)
    ).astype("float32")
    return features.astype({col: "float32" for col in features.columns if col not in ["origin_day", "item_id", "dept_id", "cat_id", "store_id", "state_id"]})


def build_training_frame(sales: pd.DataFrame, origins: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frames = []
    target_frames = []
    for origin in origins:
        x = build_features(sales, origin)
        y = pd.DataFrame(index=sales.index)
        for seg_idx, (seg_start, seg_end) in enumerate(SEGMENTS, start=1):
            y[f"seg_{seg_idx}"] = sales[dcols(origin + seg_start - 1, seg_end - seg_start + 1)].sum(axis=1)
        feature_frames.append(x)
        target_frames.append(y.astype("float32"))
    return pd.concat(feature_frames, ignore_index=True), pd.concat(target_frames, ignore_index=True)


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, lgb.LGBMRegressor]:
    models: dict[str, lgb.LGBMRegressor] = {}
    for target in y_train.columns:
        model = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=120,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.10,
            random_state=2026,
            n_jobs=args.threads,
            verbosity=-1,
        )
        model.fit(x_train, y_train[target], categorical_feature=categorical_cols)
        models[target] = model
    return models


def allocate_daily(sales: pd.DataFrame, origin: int, segment_predictions: np.ndarray) -> np.ndarray:
    daily = np.zeros((len(sales), 28), dtype=np.float32)
    for seg_idx, (seg_start, seg_end) in enumerate(SEGMENTS):
        offsets = list(range(seg_start - 1, seg_end))
        prior_cols = [f"d_{origin - 28 + offset}" for offset in offsets]
        prior = sales[prior_cols].to_numpy(dtype=np.float32)
        row_sum = prior.sum(axis=1)
        shares = np.divide(prior, row_sum[:, None], out=np.full_like(prior, 1 / len(offsets)), where=row_sum[:, None] > 0)
        daily[:, offsets] = segment_predictions[:, seg_idx][:, None] * shares
    return np.maximum(daily, 0)


def main() -> None:
    args = parse_args()
    sales = pd.read_csv(args.sales, usecols=ID_COLS + DAY_COLS)
    origins = training_origins(args.train_start, args.train_end_origin, args.step)
    x_train, y_train = build_training_frame(sales, origins)
    _, categorical_cols = base_feature_frame(sales)
    models = train_models(x_train, y_train, categorical_cols, args)

    validation_origin = 1914
    x_valid = build_features(sales, validation_origin)
    segment_preds = np.column_stack(
        [np.maximum(models[f"seg_{idx}"].predict(x_valid), 0) for idx in range(1, len(SEGMENTS) + 1)]
    )
    daily = allocate_daily(sales, validation_origin, segment_preds)

    out = pd.DataFrame(daily, columns=FORECAST_COLS)
    out.insert(0, "id", sales["id"].str.replace("_evaluation", "_validation", regex=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    feature_importance = []
    for target, model in models.items():
        for name, value in zip(x_train.columns, model.feature_importances_):
            feature_importance.append({"target": target, "feature": name, "importance": int(value)})
    fi_path = args.out.with_name("direct_segment_lightgbm_feature_importance.csv")
    pd.DataFrame(feature_importance).to_csv(fi_path, index=False)

    summary = {
        "train_origins": origins,
        "train_rows": int(len(x_train)),
        "features": int(x_train.shape[1]),
        "targets": list(y_train.columns),
        "prediction_file": str(args.out),
        "feature_importance_file": str(fi_path),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {fi_path}")
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
