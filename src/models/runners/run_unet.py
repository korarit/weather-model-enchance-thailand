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
    get_ablation_features,
    generate_smoke_test_dataset,
    export_predictions,
    assert_no_dwr_leakage,
    assert_valid_training_years,
    resolve_runner_execution_targets,
    standardize_dataframe_columns,
    load_training_dataset,
    get_train_val_split,
    numpy_to_torch,
    torch_to_numpy,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class UNetResidualBlock(nn.Module):
    """Residual Dense Block with LayerNorm and GELU for high-fidelity representation."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return self.act(h + res)


class SpatialUNetModel(nn.Module):
    """
    Contracting-Expanding U-Net Architecture with Skip Connections.
    Encodes multi-modal meteorological & topographic features through a contraction bottleneck,
    then progressively reconstructs via decoding layers fused with skip connections from
    the encoder pathway.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        # Encoder (Contracting pathway)
        h1_dim = hidden_dim * 2
        h2_dim = hidden_dim
        b_dim = max(16, hidden_dim // 2)

        self.enc1 = UNetResidualBlock(in_features, h1_dim)
        self.enc2 = UNetResidualBlock(h1_dim, h2_dim)
        self.bottleneck = UNetResidualBlock(h2_dim, b_dim)

        # Decoder (Expanding pathway with skip connections)
        self.dec2 = UNetResidualBlock(b_dim + h2_dim, h2_dim)
        self.dec1 = UNetResidualBlock(h2_dim + h1_dim, h1_dim)

        # Output regression head
        self.out_head = nn.Sequential(
            nn.Linear(h1_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, in_features]
        e1 = self.enc1(x)                                 # Contracting Stage 1
        e2 = self.enc2(e1)                                # Contracting Stage 2
        b = self.bottleneck(e2)                           # Bottleneck representation

        # Expanding pathway with U-Net skip connection concatenation
        d2 = self.dec2(torch.cat([b, e2], dim=-1))        # Fuse bottleneck + skip e2
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))       # Fuse d2 + skip e1
        return self.out_head(d1).squeeze(-1)


def train_and_eval_unet(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False
):
    """Trains Convolutional U-Net on feature representations and exports predictions."""
    import gc
    features = get_ablation_features(df, ablation)
    logger.info("Training Spatial U-Net on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    # Strict Out-Of-Time Partitioning: Train on 2021-2023, strictly lock 2024 as Validation
    train_mask, val_mask = get_train_val_split(df, val_year=2024)

    # Feature normalization to prevent large magnitude distortion (CAPE/Pressure vs Small features)
    x_train_raw = df.loc[train_mask, features].fillna(0.0).to_numpy(dtype=np.float32)
    feat_mean = np.nanmean(x_train_raw, axis=0, keepdims=True)
    feat_std = np.nanstd(x_train_raw, axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0

    x_train_norm = np.nan_to_num((x_train_raw - feat_mean) / feat_std, nan=0.0)
    x_val_raw = df.loc[val_mask, features].fillna(0.0).to_numpy(dtype=np.float32)
    x_val_norm = np.nan_to_num((x_val_raw - feat_mean) / feat_std, nan=0.0)

    X_train_t = numpy_to_torch(x_train_norm)
    y_train_t = numpy_to_torch(df.loc[train_mask, "target_bias"].to_numpy(dtype=np.float32))
    X_val_t = numpy_to_torch(x_val_norm)
    y_val_t = numpy_to_torch(df.loc[val_mask, "target_bias"].to_numpy(dtype=np.float32))

    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.loc[val_mask, meta_cols].copy()

    model = SpatialUNetModel(in_features=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    batch_size = 16 if smoke_test else min(512, max(64, len(X_train_t) // 20))
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epochs = 2 if smoke_test else 15
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        logger.info("  -> [U-Net Epoch %d/%d] Training Huber Loss: %.4f", ep, epochs, avg_loss)

    # Free training tensors
    del X_train_t, y_train_t, train_dataset, train_loader
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
        val_pred_bias = torch_to_numpy(model(X_val_t)).flatten()

    y_val_arr = torch_to_numpy(y_val_t).flatten()
    nwp_raw_val = df_val["nwp_rain_raw"].to_numpy(dtype=float) if "nwp_rain_raw" in df_val.columns else np.zeros(len(df_val))
    obs_val = df_val["observed_rain"].to_numpy(dtype=float) if "observed_rain" in df_val.columns else (nwp_raw_val + y_val_arr)
    raw_rmse = float(np.sqrt(np.mean((nwp_raw_val - obs_val) ** 2)))
    corrected_val = np.maximum(0.0, nwp_raw_val + val_pred_bias)
    val_rmse = float(np.sqrt(np.mean((corrected_val - obs_val) ** 2)))
    diff_pct = ((raw_rmse - val_rmse) / raw_rmse) * 100.0 if raw_rmse > 0 else 0.0

    logger.info(
        "U-Net [%s] [%s] Val RMSE -> Before (Raw NWP): %.4f | After (Corrected): %.4f (Skill: %+.2f%%) [on %d val samples]",
        weather_model.upper(), ablation.upper(), raw_rmse, val_rmse, diff_pct, len(y_val_arr)
    )

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
    parser.add_argument("--weather-model", type=str, default=None, help="Target Weather Model (e.g. 'ecmwf_ifs', 'gfs', 'ecmwf_ifs,gfs', or 'all')")
    parser.add_argument("--ablation", type=str, default=None, help="Feature ablation variant (e.g. 'm1', 'm2', 'm3', 'm1,m2', or 'all')")
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
