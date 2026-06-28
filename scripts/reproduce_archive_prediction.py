"""Reproduce the archived M5 LightGBM validation forecast.

This script uses the pre-generated ``archive/test_*.pkl`` feature frames and
``archive/lgb_model_*_v1.bin`` pickled LightGBM boosters that are already in
this workspace. It does not retrain models.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "sales"
END_TRAIN = 1913
PREDICTION_HORIZON = 28
ROLLING_SPECS = [(shift, window) for shift in (1, 7, 14) for window in (7, 14, 30, 60)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", default="archive", type=Path)
    parser.add_argument("--sample-submission", default="sample_submission.csv", type=Path)
    parser.add_argument("--predictions-out", default="output/archive_validation_predictions.csv", type=Path)
    parser.add_argument("--submission-out", default="output/submission_v1_local.csv", type=Path)
    parser.add_argument("--model-version", default=1, type=int)
    parser.add_argument("--clip-zero", action="store_true", help="Clip negative forecasts to 0.")
    return parser.parse_args()


def store_id_from_test_path(path: Path) -> str:
    return path.stem.replace("test_", "")


def load_base_test(archive_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(archive_dir.glob("test_*.pkl")):
        store_id = store_id_from_test_path(path)
        frame = pd.read_pickle(path)
        frame["store_id"] = store_id
        frames.append(frame)
        print(f"loaded {path.name}: {frame.shape}")

    if not frames:
        raise FileNotFoundError(f"No test_*.pkl files found in {archive_dir}")

    base_test = pd.concat(frames, ignore_index=True)
    print(f"base_test: {base_test.shape}, d={base_test['d'].min()}..{base_test['d'].max()}")
    return base_test


def load_models(archive_dir: Path, stores: list[str], model_version: int) -> dict[str, object]:
    models = {}
    for store_id in stores:
        model_path = archive_dir / f"lgb_model_{store_id}_v{model_version}.bin"
        with model_path.open("rb") as handle:
            models[store_id] = pickle.load(handle)
        print(f"loaded {model_path.name}")
    return models


def add_dynamic_rolling_features(base_test: pd.DataFrame) -> pd.DataFrame:
    grid_df = base_test.copy()
    grouped_sales = base_test.groupby("id", observed=True)[TARGET]

    rolling_features = []
    for shift_day, rolling_window in ROLLING_SPECS:
        col_name = f"rolling_mean_tmp_{shift_day}_{rolling_window}"
        values = grouped_sales.transform(
            lambda series: series.shift(shift_day).rolling(rolling_window).mean()
        )
        rolling_features.append(values.astype(np.float16).rename(col_name))

    return pd.concat([grid_df] + rolling_features, axis=1)


def predict(args: argparse.Namespace) -> pd.DataFrame:
    base_test = load_base_test(args.archive_dir)
    stores = sorted(base_test["store_id"].unique())
    models = load_models(args.archive_dir, stores, args.model_version)
    feature_columns = models[stores[0]].feature_name()

    missing_features = sorted(set(feature_columns) - set(base_test.columns) - {
        f"rolling_mean_tmp_{shift}_{window}" for shift, window in ROLLING_SPECS
    })
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    all_preds: pd.DataFrame | None = None
    main_start = time.time()

    for day in range(1, PREDICTION_HORIZON + 1):
        day_start = time.time()
        predict_d = END_TRAIN + day
        print(f"Predict day {day:02d} / d_{predict_d}")

        grid_df = add_dynamic_rolling_features(base_test)
        day_mask = base_test["d"].eq(predict_d)

        for store_id, model in models.items():
            mask = day_mask & base_test["store_id"].eq(store_id)
            pred = model.predict(grid_df.loc[mask, feature_columns])
            if args.clip_zero:
                pred = np.clip(pred, 0, None)
            base_test.loc[mask, TARGET] = pred

        temp = base_test.loc[day_mask, ["id", TARGET]].copy()
        temp.columns = ["id", f"F{day}"]
        all_preds = temp if all_preds is None else all_preds.merge(temp, on="id", how="left")

        elapsed = (time.time() - day_start) / 60
        total_elapsed = (time.time() - main_start) / 60
        print(
            f"day {day:02d}: {elapsed:.2f} min, total {total_elapsed:.2f} min, "
            f"forecast sales={temp[f'F{day}'].sum():.2f}"
        )

    assert all_preds is not None
    return all_preds


def write_outputs(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(args.predictions_out, index=False)
    print(f"wrote {args.predictions_out} ({predictions.shape})")

    sample = pd.read_csv(args.sample_submission)[["id"]]
    submission = sample.merge(predictions, on="id", how="left").fillna(0)
    submission.to_csv(args.submission_out, index=False)
    print(f"wrote {args.submission_out} ({submission.shape})")


def main() -> None:
    args = parse_args()
    predictions = predict(args)
    write_outputs(predictions, args)


if __name__ == "__main__":
    main()
