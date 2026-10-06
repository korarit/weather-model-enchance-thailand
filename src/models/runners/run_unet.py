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
    features = ABLATION_FEATURES[ablation]
    logger.info("Training Spatial U-Net on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    X_mat = df[features].copy().fillna(0.0).to_numpy(dtype=np.float32)
    y_vec = df["target_bias"].to_numpy(dtype=np.float32)

    split_idx = int(len(df) * 0.7)
    X_train_t = torch.tensor(X_mat[:split_idx])
    y_train_t = torch.tensor(y_vec[:split_idx])
    X_val_t = torch.tensor(X_mat[split_idx:])
    y_val_t = torch.tensor(y_vec[split_idx:])
    df_val = df.iloc[split_idx:].copy()

    model = SpatialUNetModel(in_features=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 2 if smoke_test else 20
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

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

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="unet",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )


def main():
    parser = argparse.ArgumentParser(description="Spatial U-Net Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", type=str, default="outputs/unet/", help="Base output directory")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--data-file", type=str, help="Path to training features parquet")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.dir
    weather_models, ablations, hyperparams = resolve_runner_execution_targets(
        config_path=args.config,
        weather_model_arg=args.weather_model,
        ablation_arg=args.ablation,
    )

    logger.info("U-Net Runner Targets -> Weather Models: %s | Ablations: %s", weather_models, ablations)

    for wm in weather_models:
        if args.data_file:
            assert_no_dwr_leakage(args.data_file)
            df = pd.read_parquet(args.data_file)
            assert_valid_training_years(df)
        else:
            df = generate_smoke_test_dataset(n_samples=100, weather_model=wm)

        for ab in ablations:
            train_and_eval_unet(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
