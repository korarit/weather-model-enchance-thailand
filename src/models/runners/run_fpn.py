"""
Feature Pyramid Network (FPN) Spatial Bias Correction Runner (Decoupled Pipeline)
Multi-Scale Feature Hierarchies with Top-Down Lateral Pyramidal Fusion
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
import torch
import torch.nn as nn
import torch.optim as optim

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


class SpatialFPNModel(nn.Module):
    """
    Feature Pyramid Network (FPN) for Spatial Weather Bias Correction.
    Key properties:
    - Bottom-up hierarchical feature extraction (C1 -> C2 -> C3).
    - Top-down pathway with 1x1 lateral projection connections (P3 -> P2 -> P1).
    - Multi-scale pyramidal feature fusion across resolution levels,
      capturing both fine-scale topography and synoptic atmospheric flows.
    """
    def __init__(self, in_features: int, hidden_dim: int = 32, pyramid_channels: int = 24):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)

        # Bottom-up pathway (Encoders at progressive scales)
        self.c1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Lateral 1x1 convolutions to uniform pyramid channel depth
        self.lat_c3 = nn.Conv1d(64, pyramid_channels, kernel_size=1)
        self.lat_c2 = nn.Conv1d(32, pyramid_channels, kernel_size=1)
        self.lat_c1 = nn.Conv1d(16, pyramid_channels, kernel_size=1)

        # Anti-aliasing smooth convolutions for pyramid feature maps
        self.smooth_p3 = nn.Conv1d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv1d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1)
        self.smooth_p1 = nn.Conv1d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1)

        # Multi-scale pyramid fusion head
        self.fusion = nn.Sequential(
            nn.Conv1d(pyramid_channels * 3, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.out_head = nn.Sequential(
            nn.Linear(32 * hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, in_features]
        h = torch.relu(self.in_proj(x)).unsqueeze(1)  # [Batch, 1, hidden_dim]

        # 1. Bottom-up pathway
        c1 = self.c1(h)                               # [Batch, 16, hidden_dim]
        c2 = self.c2(c1)                              # [Batch, 32, hidden_dim]
        c3 = self.c3(c2)                              # [Batch, 64, hidden_dim]

        # 2. Top-down pathway with lateral connections
        p3 = self.lat_c3(c3)                          # [Batch, pyramid_channels, hidden_dim]
        p2 = self.lat_c2(c2) + p3                     # Lateral fusion
        p1 = self.lat_c1(c1) + p2                     # Lateral fusion

        # 3. Smoothing
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)
        p1 = self.smooth_p1(p1)

        # 4. Multi-scale feature concatenation & projection
        pyramid_fused = torch.cat([p1, p2, p3], dim=1) # [Batch, pyramid_channels * 3, hidden_dim]
        fused = self.fusion(pyramid_fused)            # [Batch, 32, hidden_dim]

        flat = fused.view(fused.size(0), -1)
        return self.out_head(flat).squeeze(-1)


def train_and_eval_fpn(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False
):
    """Trains Feature Pyramid Network (FPN) on multi-modal features and exports predictions."""
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Spatial FPN on [%s] [%s] with %d features (smoke_test=%s)...",
                weather_model.upper(), ablation.upper(), len(features), smoke_test)

    # Strict Out-Of-Time Partitioning: Train on 2021-2023, strictly lock 2024 as Validation
    train_mask, val_mask = get_train_val_split(df, val_year=2024)

    X_train_t = torch.from_numpy(df.loc[train_mask, features].fillna(0.0).to_numpy(dtype=np.float32))
    y_train_t = torch.from_numpy(df.loc[train_mask, "target_bias"].to_numpy(dtype=np.float32))
    X_val_t = torch.from_numpy(df.loc[val_mask, features].fillna(0.0).to_numpy(dtype=np.float32))
    y_val_t = torch.from_numpy(df.loc[val_mask, "target_bias"].to_numpy(dtype=np.float32))

    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.loc[val_mask, meta_cols].copy()


    model = SpatialFPNModel(in_features=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 2 if smoke_test else 20
    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        logger.info("  -> [FPN Epoch %d/%d] Training Huber Loss: %.4f", ep, epochs, loss.item())

    # Free training tensors
    del X_train_t, y_train_t
    gc.collect()

    # Save checkpoint
    model_dir = output_dir / "models" / "fpn" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_file)
    logger.info("Saved FPN checkpoint: %s", ckpt_file)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred_bias = model(X_val_t).numpy()

    y_val_arr = y_val_t.numpy()
    raw_rmse = float(np.sqrt(np.mean(y_val_arr ** 2)))
    val_rmse = float(np.sqrt(np.mean((val_pred_bias - y_val_arr) ** 2)))
    diff_pct = ((raw_rmse - val_rmse) / raw_rmse) * 100.0 if raw_rmse > 0 else 0.0

    logger.info(
        "FPN [%s] [%s] Val RMSE -> Before (Raw NWP): %.4f | After (Corrected): %.4f (Skill: %+.2f%%) [on %d val samples]",
        weather_model.upper(), ablation.upper(), raw_rmse, val_rmse, diff_pct, len(y_val_arr)
    )

    del X_val_t, y_val_t
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="fpn",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )

    del df_val, val_pred_bias, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Spatial Feature Pyramid Network (FPN) Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/fpn/", help="Base output directory")
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
    logger.info("FPN Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Spatial FPN | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_fpn(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
