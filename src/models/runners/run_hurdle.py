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
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_and_eval_hurdle(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    rain_threshold: float = 0.1,
    prob_threshold: float = 0.35,
    smoke_test: bool = False,
    hyperparams: dict = None,
):
    """Trains Two-Stage Hurdle Model and exports predictions."""
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Two-Stage Hurdle on [%s] [%s] (smoke_test=%s)...", weather_model.upper(), ablation.upper(), smoke_test)

    X = df[features].copy().fillna(0.0)
    y_rain_binary = (df["observed_rain"] >= rain_threshold).astype(int)
    y_bias = df["target_bias"].copy()

    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_bin_train, y_bin_val = y_rain_binary.iloc[:split_idx], y_rain_binary.iloc[split_idx:]
    y_bias_train, y_bias_val = y_bias.iloc[:split_idx], y_bias.iloc[split_idx:]
    df_val = df.iloc[split_idx:].copy()

    n_estimators = 2 if smoke_test else 80

    # STAGE 1: Occurrence Classifier
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=15 if smoke_test else 31,
        learning_rate=0.08,
        verbose=-1,
        random_state=42
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
        "random_state": 42
    }
    if hyperparams and not smoke_test:
        reg_params.update(hyperparams)

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(X_train[pos_mask_train], y_bias_train[pos_mask_train])

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

    export_predictions(
        df=df_val,
        predicted_bias=final_bias_pred,
        model_name="hurdle",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Hurdle GBDT Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/hurdle/", help="Base output directory")
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
    logger.info("Hurdle Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
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
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Hurdle GBDT | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_hurdle(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
