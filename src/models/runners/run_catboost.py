"""
CatBoost Bias Correction Runner (Decoupled Pipeline)
Supports Oblivious Trees with RMSE / Tweedie Loss and Ablations M1, M2, M3
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
from catboost import CatBoostRegressor

from src.models.common import (
    ABLATION_FEATURES,
    generate_smoke_test_dataset,
    export_predictions,
    assert_no_dwr_leakage,
    assert_valid_training_years,
    resolve_runner_execution_targets,
    standardize_dataframe_columns,
    load_training_dataset,
    get_train_val_split,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_and_eval_catboost(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False,
    hyperparams: dict = None,
):
    """Trains CatBoost regressor on residual bias and exports predictions."""
    import gc
    features = ABLATION_FEATURES[ablation]
    logger.info("Training CatBoost on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    # Strict Out-Of-Time Partitioning: Train on 2021-2023, strictly lock 2024 as Validation
    train_mask, val_mask = get_train_val_split(df, val_year=2024)

    # Train set (Years 2021-2023 or chronologically first 70%)
    X_train = df.loc[train_mask, features].fillna(0.0)
    y_train = df.loc[train_mask, "target_bias"]

    # Validation set (Strictly Year 2024 or chronologically last 30%)
    X_val = df.loc[val_mask, features].fillna(0.0)
    y_val = df.loc[val_mask, "target_bias"]

    # Extract only required metadata columns for prediction export (reduces val slice RAM by 80%)
    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.loc[val_mask, meta_cols].copy()


    iterations = 3 if smoke_test else 100
    params = {
        "iterations": iterations,
        "depth": 4 if smoke_test else 6,
        "learning_rate": 0.08,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    if hyperparams and not smoke_test:
        params.update(hyperparams)

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Free heavy training matrix immediately after fit to relieve RAM
    del X_train, y_train
    gc.collect()

    # Save model checkpoint
    model_dir = output_dir / "models" / "catboost" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "model.cbm"
    model.save_model(str(ckpt_file))
    logger.info("Saved CatBoost checkpoint: %s", ckpt_file)

    val_pred_bias = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((val_pred_bias - y_val.to_numpy()) ** 2)))
    logger.info("CatBoost [%s] [%s] Validation RMSE: %.4f (on %d val samples)",
                weather_model.upper(), ablation.upper(), val_rmse, len(y_val))

    # Free X_val, y_val before export
    del X_val, y_val
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="catboost",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )

    del df_val, val_pred_bias, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="CatBoost Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/catboost/", help="Base output directory")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--data-file", "--data-dir", "--data-path", dest="data_file", type=str, default=None, help="Path to training features parquet file or directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum training samples to load (recommended: 500000 or 1000000 on Colab 12GB RAM)")
    args = parser.parse_args()

    dir_p = Path(args.dir)
    output_dir = dir_p if dir_p.is_absolute() else PROJECT_ROOT / dir_p
    weather_models, ablations, hyperparams = resolve_runner_execution_targets(
        config_path=args.config,
        weather_model_arg=args.weather_model,
        ablation_arg=args.ablation,
    )

    total_runs = len(weather_models) * len(ablations)
    logger.info("CatBoost Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: CatBoost | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_catboost(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
