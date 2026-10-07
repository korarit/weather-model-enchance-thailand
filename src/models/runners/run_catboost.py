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
    features = ABLATION_FEATURES[ablation]
    logger.info("Training CatBoost on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    X = df[features].copy().fillna(0.0)
    y = df["target_bias"].copy()

    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    df_val = df.iloc[split_idx:].copy()

    iterations = 3 if smoke_test else 100
    params = {
        "iterations": iterations,
        "depth": 4 if smoke_test else 6,
        "learning_rate": 0.08,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0
    }
    if hyperparams and not smoke_test:
        params.update(hyperparams)

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

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

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="catboost",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )


def main():
    parser = argparse.ArgumentParser(description="CatBoost Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/catboost/", help="Base output directory")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--data-file", type=str, help="Path to training features parquet")
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
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: CatBoost | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_catboost(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
