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
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Multi-Quantile GBDT on [%s] [%s] for %s (smoke_test=%s)...", weather_model.upper(), ablation.upper(), QUANTILES, smoke_test)

    X = df[features].copy().fillna(0.0)
    y = df["target_bias"].copy()

    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    df_val = df.iloc[split_idx:].copy()

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
            "random_state": 42
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

    logger.info("Saved all quantile checkpoints in %s", model_dir)

    pred_q50 = median_pred_bias if median_pred_bias is not None else quantile_preds["q50"]
    val_rmse_q50 = float(np.sqrt(np.mean((pred_q50 - y_val.to_numpy()) ** 2)))
    logger.info("Multi-Quantile [%s] [%s] q50 Validation RMSE: %.4f", weather_model.upper(), ablation.upper(), val_rmse_q50)

    export_predictions(
        df=df_val,
        predicted_bias=pred_q50,
        model_name="quantile",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
        quantile_preds=quantile_preds
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-Quantile GBDT Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", type=str, default="outputs/quantile/", help="Base output directory")
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
    logger.info("Quantile Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        if args.data_file:
            assert_no_dwr_leakage(args.data_file)
            df = pd.read_parquet(args.data_file)
            assert_valid_training_years(df)
        else:
            df = generate_smoke_test_dataset(n_samples=100, weather_model=wm)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Multi-Quantile | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_quantile(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test, hyperparams=hyperparams)


if __name__ == "__main__":
    main()
