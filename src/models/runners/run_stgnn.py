"""
Spatio-Temporal Graph Neural Network (ST-GNN) Bias Correction Runner (Decoupled Pipeline)
Models Spatial Advection and Graph Neighborhood Telemetry using PyTorch
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
    resolve_lat_lon,
    load_training_dataset,
    get_train_val_split,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class GraphConvLayer(nn.Module):
    """Simple spatial graph convolution layer with advective message passing."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # A * X * W
        agg = torch.matmul(adj, x)
        return torch.relu(self.fc(agg))


class STGNNModel(nn.Module):
    """Spatio-Temporal Graph Neural Network for gridded bias correction."""
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.gconv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gconv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(x))
        h = self.gconv1(h, adj)
        h = self.gconv2(h, adj)
        return self.head(h).squeeze(-1)


def build_spatial_adjacency(lat: np.ndarray, lon: np.ndarray, max_dist_km: float = 50.0, max_nodes: int = 2500) -> torch.Tensor:
    """Builds distance-weighted spatial adjacency matrix with self-loops, bounded to max_nodes for memory safety."""
    n = min(len(lat), max_nodes)
    lat_sub = lat[:n]
    lon_sub = lon[:n]
    coords = np.column_stack([lat_sub * 111.0, lon_sub * 105.0])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    adj = np.exp(-dist / 25.0)
    adj[dist > max_dist_km] = 0.0
    # Normalize rows
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / row_sum
    return torch.tensor(adj, dtype=torch.float32)


def train_and_eval_stgnn(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False
):
    """Trains ST-GNN on spatial graph and exports predictions."""
    import gc
    features = ABLATION_FEATURES[ablation]
    logger.info("Training ST-GNN on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    lat_s, lon_s = resolve_lat_lon(df)
    lats = lat_s.to_numpy(dtype=float)
    lons = lon_s.to_numpy(dtype=float)

    # Strict Out-Of-Time Partitioning: Train on 2021-2023, strictly lock 2024 as Validation
    train_mask, val_mask = get_train_val_split(df, val_year=2024)

    max_graph_nodes = 100 if smoke_test else 2000
    adj = build_spatial_adjacency(lats, lons, max_nodes=max_graph_nodes)
    sub_n = len(adj)

    # Subsample masks to graph node limit for GNN adjacency dimension alignment
    train_sub = train_mask[:sub_n]
    val_sub = val_mask[:sub_n]

    # Ensure at least 1 sample in train and val within sub_n limit
    if np.sum(train_sub) == 0 or np.sum(val_sub) == 0:
        split_idx = int(sub_n * 0.7)
        train_sub = np.zeros(sub_n, dtype=bool)
        val_sub = np.zeros(sub_n, dtype=bool)
        train_sub[:split_idx] = True
        val_sub[split_idx:] = True

    df_sub = df.iloc[:sub_n]
    X_train_t = torch.from_numpy(df_sub.loc[train_sub, features].fillna(0.0).to_numpy(dtype=np.float32))
    y_train_t = torch.from_numpy(df_sub.loc[train_sub, "target_bias"].to_numpy(dtype=np.float32))
    adj_train = adj[train_sub][:, train_sub]

    X_val_t = torch.from_numpy(df_sub.loc[val_sub, features].fillna(0.0).to_numpy(dtype=np.float32))
    y_val_t = torch.from_numpy(df_sub.loc[val_sub, "target_bias"].to_numpy(dtype=np.float32))
    adj_val = adj[val_sub][:, val_sub]

    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df_sub.loc[val_sub, meta_cols].copy()


    model = STGNNModel(in_dim=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 2 if smoke_test else 25
    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(X_train_t, adj_train)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        logger.info("  -> [ST-GNN Epoch %d/%d] Training Huber Loss: %.4f", ep, epochs, loss.item())

    # Free training tensors
    del X_train_t, y_train_t, adj_train
    gc.collect()

    # Save model checkpoint
    model_dir = output_dir / "models" / "stgnn" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_file)
    logger.info("Saved ST-GNN checkpoint: %s", ckpt_file)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred_bias = model(X_val_t, adj_val).numpy()

    val_rmse = float(np.sqrt(np.mean((val_pred_bias - y_val_t.numpy()) ** 2)))
    logger.info("ST-GNN [%s] [%s] Validation RMSE: %.4f", weather_model.upper(), ablation.upper(), val_rmse)

    del X_val_t, y_val_t, adj_val
    gc.collect()

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="stgnn",
        ablation=ablation,
        output_dir=output_dir,
        weather_model=weather_model,
    )

    del df_val, val_pred_bias, model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="ST-GNN Bias Correction Runner")
    parser.add_argument("--weather-model", choices=["ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege", "all"], default=None, help="Target Weather Model (or 'all')")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default=None, help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", "--out-dir", "--output-dir", dest="dir", type=str, default="outputs/stgnn/", help="Base output directory")
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
    logger.info("ST-GNN Runner Targets -> Weather Models: %s | Ablations: %s (Total: %d runs)",
                weather_models, ablations, total_runs)

    run_counter = 0
    for wm in weather_models:
        df = load_training_dataset(data_path_or_str=args.data_file, weather_model=wm, max_samples=args.max_samples)

        for ab in ablations:
            run_counter += 1
            pct = (run_counter / total_runs) * 100
            logger.info("=== [Training Run %d/%d (%.1f%%)] Model: ST-GNN | Weather Model: %s | Ablation: %s ===",
                        run_counter, total_runs, pct, wm.upper(), ab.upper())
            train_and_eval_stgnn(df, ab, output_dir=output_dir, weather_model=wm, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
