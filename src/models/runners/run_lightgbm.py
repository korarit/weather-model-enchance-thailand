"""
LightGBM Bias Correction Runner (Decoupled Pipeline)
Supports Ablations: M1 (Ground), M2 (Satellite), M3 (Joint Ground+Satellite)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.common import (
    ABLATION_FEATURES,
    generate_smoke_test_dataset,
    export_predictions,
    assert_no_dwr_leakage,
    assert_valid_training_years,
    resolve_runner_execution_targets,
    standardize_dataframe_columns,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_and_eval_lgbm(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False,
    hyperparams: dict = None,
):
    """Trains LightGBM regressor on residual bias and exports predictions."""
    features = ABLATION_FEATURES[ablation]
    logger.info("Training LightGBM on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    X = df[features].copy()
    y = df["target_bias"].copy()

    # Handle missing values
    X = X.fillna(0.0)

    # Train / Val Split (in smoke test, first 70% train, last 30% val)
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    df_val = df.iloc[split_idx:].copy()

    n_estimators = 2 if smoke_test else 100
    params = {
        "objective": "regression_l1",  # MAE / Huber style robust to extremes
        "metric": "l1",
        "num_leaves": 15 if smoke_test else 31,
        "learning_rate": 0.05,
        "n_estimators": n_estimators,
        "verbose": -1,
        "random_state": 42,
    }
    if hyperparams and not smoke_test:
        params.update(hyperparams)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(5, verbose=False)] if not smoke_test else None)

    # Save model checkpoint
    model_dir = output_dir / "models" / "lightgbm" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "model.txt"
    model.booster_.save_model(str(ckpt_file))
    logger.info("Saved LightGBM checkpoint: %s", ckpt_file)

    # Predict bias and export
    val_pred_bias = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((val_pred_bias - y_val.to_numpy()) ** 2)))
    logger.info("LightGBM [%s] [%s] Validation RMSE: %.4f (on %d val samples)",
                weather_model.upper(), ablation.upper(), val_rmse, len(y_val))

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="lightgbm",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )


def main():
    parser = argparse.ArgumentParser(description="LightGBM Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC (50-100 rows, 1-2 trees)")
    parser.add_argument("--dir", type=str, default="outputs/lightgbm/", help="Base output directory")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--data-file", type=str, help="Path to training features parquet")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.dir
    weather_models, ablations, hyperparams = resolve_runner_execution_targets(
        config_path=args.config,
        weather_model_arg=args.weather_model,
        ablation_arg=args.ablation,
    )

    total_runs = len(weather_models) * len(ablations)
    logger.info("LightGBM Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        if args.data_file:
            assert_no_dwr_leakage(args.data_file)
            df = pd.read_parquet(args.data_file)
            assert_valid_training_years(df)
            df = standardize_dataframe_columns(df)
        else:
            df = generate_smoke_test_dataset(n_samples=100, weather_model=wm)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: LightGBM | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_lgbm(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()

