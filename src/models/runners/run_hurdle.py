"""
Two-Stage Hurdle GBDT Bias Correction Runner (Decoupled Pipeline)
Solves Drizzle Bias and Zero-Inflation:
- Stage 1: Rain Occurrence Classifier P(Rain >= 0.1mm)
- Stage 2: Conditional Residual Bias Regressor
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


def train_and_eval_hurdle(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    rain_threshold: float = 0.1,
    prob_threshold: float = 0.5,
    smoke_test: bool = False,
    hyperparams: dict = None,
):
    """Trains Two-Stage Hurdle Model and exports predictions."""
    import gc
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Two-Stage Hurdle on [%s] [%s] (smoke_test=%s)...", weather_model.upper(), ablation.upper(), smoke_test)

    # ZERO-COPY slicing directly on df to avoid duplicating multi-gigabyte matrices
    split_idx = int(len(df) * 0.7)
    
    # Train set (70%)
    X_train = df.iloc[:split_idx][features].fillna(0.0)
    y_bin_train = (df.iloc[:split_idx]["observed_rain"] >= rain_threshold).astype(int)
    y_bias_train = df.iloc[:split_idx]["target_bias"]

    # Validation set (30%)
    X_val = df.iloc[split_idx:][features].fillna(0.0)
    y_bin_val = (df.iloc[split_idx:]["observed_rain"] >= rain_threshold).astype(int)
    y_bias_val = df.iloc[split_idx:]["target_bias"]

    # Extract only required metadata columns for prediction export (reduces val slice RAM by 80%)
    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.iloc[split_idx:][meta_cols].copy()

    n_estimators = 2 if smoke_test else 80

    # STAGE 1: Occurrence Classifier
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=15 if smoke_test else 31,
        learning_rate=0.08,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_bin_train)

    # STAGE 2: Conditional Bias Regressor (trained only on positive rain instances)
    pos_mask_train = y_bin_train == 1
    if pos_mask_train.sum() < 2:
        pos_mask_train = np.ones(len(y_bin_train), dtype=bool)

    reg_params = {
        "objective": "regression_l1",
        "n_estimators": n_estimators,
        "num_leaves": 15 if smoke_test else 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if hyperparams and not smoke_test:
        reg_params.update(hyperparams)

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(X_train[pos_mask_train], y_bias_train[pos_mask_train])

    # Free training matrices immediately
    del X_train, y_bin_train, y_bias_train, pos_mask_train
    gc.collect()

    # Save checkpoints
    model_dir = output_dir / "models" / "hurdle" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(model_dir / "clf.txt"))
    reg.booster_.save_model(str(model_dir / "reg.txt"))
    logger.info("Saved Hurdle checkpoints in %s", model_dir)

    # Inference: Combined Decision
    probs = clf.predict_proba(X_val)[:, 1]
    bias_preds_cond = reg.predict(X_val)
    
    # If prob < threshold -> predicted_rain = 0, meaning predicted_bias = -nwp_raw
    nwp_raw_val = df_val["nwp_rain_raw"].to_numpy(dtype=float)
    final_bias_pred = np.where(probs >= prob_threshold, bias_preds_cond, -nwp_raw_val)

    val_rmse = float(np.sqrt(np.mean((final_bias_pred - y_bias_val.to_numpy()) ** 2)))
    logger.info("Hurdle [%s] [%s] Combined Validation RMSE: %.4f", weather_model.upper(), ablation.upper(), val_rmse)

    del X_val, y_bin_val, y_bias_val, probs, bias_preds_cond
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=final_bias_pred,
        model_name="hurdle",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )

    del df_val, final_bias_pred, clf, reg
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Hurdle GBDT Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/hurdle/", help="Base output directory")
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
    logger.info("Hurdle Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Hurdle GBDT | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_hurdle(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
