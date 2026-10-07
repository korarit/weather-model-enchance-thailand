"""
2D Convolutional U-Net / Spatial Downscaling Bias Correction Runner (Decoupled Pipeline)
Models Spatial Raster Tensors and Multi-channel Grid Inversion using PyTorch
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
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class UNet1DBlock(nn.Module):
    """Convolutional Residual Block for 1D/2D Feature Downscaling."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialUNetModel(nn.Module):
    """Lightweight Convolutional U-Net Architecture for tabular gridded bias correction."""
    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.enc1 = UNet1DBlock(1, 16)
        self.enc2 = UNet1DBlock(16, 32)
        self.dec1 = UNet1DBlock(32, 16)
        self.out_head = nn.Sequential(
            nn.Linear(16 * hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, in_features]
        h = torch.relu(self.in_proj(x)).unsqueeze(1)  # [Batch, 1, hidden_dim]
        e1 = self.enc1(h)                              # [Batch, 16, hidden_dim]
        e2 = self.enc2(e1)                             # [Batch, 32, hidden_dim]
        d1 = self.dec1(e2)                             # [Batch, 16, hidden_dim]
        flat = d1.view(d1.size(0), -1)
        return self.out_head(flat).squeeze(-1)


def train_and_eval_unet(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False
):
    """Trains Convolutional U-Net on feature representations and exports predictions."""
    import gc
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Spatial U-Net on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    split_idx = int(len(df) * 0.7)
    X_train_t = torch.from_numpy(df.iloc[:split_idx][features].fillna(0.0).to_numpy(dtype=np.float32))
    y_train_t = torch.from_numpy(df.iloc[:split_idx]["target_bias"].to_numpy(dtype=np.float32))
    X_val_t = torch.from_numpy(df.iloc[split_idx:][features].fillna(0.0).to_numpy(dtype=np.float32))
    y_val_t = torch.from_numpy(df.iloc[split_idx:]["target_bias"].to_numpy(dtype=np.float32))

    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.iloc[split_idx:][meta_cols].copy()

    model = SpatialUNetModel(in_features=len(features), hidden_dim=16 if smoke_test else 32)
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
        logger.info("  -> [U-Net Epoch %d/%d] Training Huber Loss: %.4f", ep, epochs, loss.item())

    # Free training tensors
    del X_train_t, y_train_t
    gc.collect()

    # Save checkpoint
    model_dir = output_dir / "models" / "unet" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_file)
    logger.info("Saved U-Net checkpoint: %s", ckpt_file)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred_bias = model(X_val_t).numpy()

    val_rmse = float(np.sqrt(np.mean((val_pred_bias - y_val_t.numpy()) ** 2)))
    logger.info("U-Net [%s] [%s] Validation RMSE: %.4f", weather_model.upper(), ablation.upper(), val_rmse)

    del X_val_t, y_val_t
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="unet",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )

    del df_val, val_pred_bias, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Spatial U-Net Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/unet/", help="Base output directory")
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
    logger.info("U-Net Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: Spatial U-Net | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_unet(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
