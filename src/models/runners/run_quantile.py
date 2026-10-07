"""
Multi-Quantile GBDT Uncertainty Quantification Runner (Decoupled Pipeline)
Trains separate Pinball Loss models for quantiles: q10, q50, q90, q95
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
    load_training_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QUANTILES = [0.10, 0.50, 0.90, 0.95]


def train_and_eval_quantile(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False,
    hyperparams: dict = None,
):
    """Trains multi-quantile models and exports prediction intervals."""
    import gc
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Multi-Quantile GBDT on [%s] [%s] for %s (smoke_test=%s)...", weather_model.upper(), ablation.upper(), QUANTILES, smoke_test)

    # ZERO-COPY slicing directly on df to avoid duplicating multi-gigabyte matrices
    split_idx = int(len(df) * 0.7)
    
    # Train set (70%)
    X_train = df.iloc[:split_idx][features].fillna(0.0)
    y_train = df.iloc[:split_idx]["target_bias"]

    # Validation set (30%)
    X_val = df.iloc[split_idx:][features].fillna(0.0)
    y_val = df.iloc[split_idx:]["target_bias"]

    # Extract only required metadata columns for prediction export (reduces val slice RAM by 80%)
    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.iloc[split_idx:][meta_cols].copy()

    n_estimators = 2 if smoke_test else 80
    model_dir = output_dir / "models" / "quantile" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)

    quantile_preds = {}
    median_pred_bias = None

    for q in QUANTILES:
        q_tag = f"q{int(q*100)}"
        params = {
            "objective": "quantile",
            "alpha": q,
            "n_estimators": n_estimators,
            "num_leaves": 15 if smoke_test else 31,
            "learning_rate": 0.08,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        if hyperparams and not smoke_test:
            params.update(hyperparams)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Save checkpoint
        ckpt_file = model_dir / f"{q_tag}.txt"
        model.booster_.save_model(str(ckpt_file))
        
        preds_q = model.predict(X_val)
        quantile_preds[q_tag] = preds_q
        if np.isclose(q, 0.50):
            median_pred_bias = preds_q
        del model
        gc.collect()

    del X_train, y_train
    gc.collect()

    logger.info("Saved all quantile checkpoints in %s", model_dir)

    pred_q50 = median_pred_bias if median_pred_bias is not None else quantile_preds["q50"]
    val_rmse_q50 = float(np.sqrt(np.mean((pred_q50 - y_val.to_numpy()) ** 2)))
    logger.info("Multi-Quantile [%s] [%s] q50 Validation RMSE: %.4f", weather_model.upper(), ablation.upper(), val_rmse_q50)

    del X_val, y_val
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=pred_q50,
        model_name="quantile",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
        quantile_preds=quantile_preds
    )

    del df_val, quantile_preds
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Multi-Quantile GBDT Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/quantile/", help="Base output directory")
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
    logger.info("Quantile Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Multi-Quantile | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_quantile(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
