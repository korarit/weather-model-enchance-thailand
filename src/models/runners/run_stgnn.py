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


def build_spatial_adjacency(lat: np.ndarray, lon: np.ndarray, max_dist_km: float = 50.0) -> torch.Tensor:
    """Builds distance-weighted spatial adjacency matrix with self-loops."""
    n = len(lat)
    coords = np.column_stack([lat * 111.0, lon * 105.0])
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
    smoke_test: bool = False
):
    """Trains ST-GNN on spatial graph and exports predictions."""
    features = ABLATION_FEATURES[ablation]
    logger.info("Training ST-GNN [%s] with %d features (smoke_test=%s)...", ablation.upper(), len(features), smoke_test)

    X_mat = df[features].copy().fillna(0.0).to_numpy(dtype=np.float32)
    y_vec = df["target_bias"].to_numpy(dtype=np.float32)
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)

    adj = build_spatial_adjacency(lats, lons)

    split_idx = int(len(df) * 0.7)
    X_train_t = torch.tensor(X_mat[:split_idx])
    y_train_t = torch.tensor(y_vec[:split_idx])
    adj_train = adj[:split_idx, :split_idx]

    X_val_t = torch.tensor(X_mat[split_idx:])
    y_val_t = torch.tensor(y_vec[split_idx:])
    adj_val = adj[split_idx:, split_idx:]
    df_val = df.iloc[split_idx:].copy()

    model = STGNNModel(in_dim=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 2 if smoke_test else 25
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t, adj_train)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    # Save model checkpoint
    model_dir = output_dir / "models" / "stgnn" / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_file)
    logger.info("Saved ST-GNN checkpoint: %s", ckpt_file)

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred_bias = model(X_val_t, adj_val).numpy()

    export_predictions(
        df=df_val,
        predicted_bias=val_pred_bias,
        model_name="stgnn",
        ablation=ablation,
        output_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description="ST-GNN Bias Correction Runner")
    parser.add_argument("--ablation", choices=["m1", "m2", "m3", "all"], default="m3", help="Feature ablation variant")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on PC")
    parser.add_argument("--dir", type=str, default="outputs/stgnn/", help="Base output directory")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--data-file", type=str, help="Path to training features parquet")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.dir

    if args.data_file:
        assert_no_dwr_leakage(args.data_file)
        df = pd.read_parquet(args.data_file)
        assert_valid_training_years(df)
    else:
        df = generate_smoke_test_dataset(n_samples=100)

    ablations = ["m1", "m2", "m3"] if args.ablation == "all" else [args.ablation]
    for ab in ablations:
        train_and_eval_stgnn(df, ab, output_dir=output_dir, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
