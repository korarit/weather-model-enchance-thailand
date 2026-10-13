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
    get_ablation_features,
    generate_smoke_test_dataset,
    export_predictions,
    assert_no_dwr_leakage,
    assert_valid_training_years,
    resolve_runner_execution_targets,
    standardize_dataframe_columns,
    resolve_lat_lon,
    load_training_dataset,
    get_train_val_split,
    numpy_to_torch,
    torch_to_numpy,
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
        h0 = torch.relu(self.encoder(x))
        h1 = h0 + self.gconv1(h0, adj)
        h2 = h1 + self.gconv2(h1, adj)
        return self.head(h2).squeeze(-1)


def build_spatial_adjacency_matrix(lats: np.ndarray, lons: np.ndarray, max_dist_km: float = 50.0) -> torch.Tensor:
    """Computes row-normalized spatial Gaussian adjacency matrix with self-loops."""
    coords = np.column_stack([lats * 111.0, lons * 105.0])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    adj = np.exp(-dist / 25.0)
    adj[dist > max_dist_km] = 0.0
    np.fill_diagonal(adj, 1.0)
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / row_sum
    return numpy_to_torch(adj, dtype=torch.float32)


def build_batch_block_diagonal_adjacency(
    sub_groups: list,
    lats: np.ndarray,
    lons: np.ndarray,
    max_dist_km: float = 50.0
) -> torch.Tensor:
    """
    Computes a block-diagonal row-normalized spatial Gaussian adjacency matrix.
    Each block corresponds to observations within the SAME timestamp snapshot.
    Cross-timestamp connections are strictly 0.0, preventing temporal graph corruption.
    """
    total_n = sum(len(idx) for idx in sub_groups)
    adj_full = np.zeros((total_n, total_n), dtype=np.float32)
    offset = 0
    for idx in sub_groups:
        n_g = len(idx)
        if n_g == 1:
            adj_full[offset, offset] = 1.0
        else:
            coords = np.column_stack([lats[idx] * 111.0, lons[idx] * 105.0])
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            adj_g = np.exp(-dist / 25.0)
            adj_g[dist > max_dist_km] = 0.0
            np.fill_diagonal(adj_g, 1.0)
            row_sum = adj_g.sum(axis=1, keepdims=True) + 1e-6
            adj_full[offset:offset + n_g, offset:offset + n_g] = adj_g / row_sum
        offset += n_g
    return numpy_to_torch(adj_full, dtype=torch.float32)


def train_and_eval_stgnn(
    df: pd.DataFrame,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    smoke_test: bool = False
):
    """Trains ST-GNN on spatial graph and exports predictions."""
    import gc
    features = get_ablation_features(df, ablation)
    logger.info("Training ST-GNN on [%s] [%s] with %d features (smoke_test=%s)...", weather_model.upper(), ablation.upper(), len(features), smoke_test)

    lat_s, lon_s = resolve_lat_lon(df)
    lats_all = lat_s.to_numpy(dtype=np.float32)
    lons_all = lon_s.to_numpy(dtype=np.float32)

    # Strict Out-Of-Time Partitioning: Train on 2021-2023, strictly lock 2024 as Validation
    train_mask, val_mask = get_train_val_split(df, val_year=2024)

    # Feature normalization to prevent magnitude distortion across heterogeneous variables
    x_train_raw = df.loc[train_mask, features].fillna(0.0).to_numpy(dtype=np.float32)
    feat_mean = np.nanmean(x_train_raw, axis=0, keepdims=True)
    feat_std = np.nanstd(x_train_raw, axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0

    x_train_norm = np.nan_to_num((x_train_raw - feat_mean) / feat_std, nan=0.0)
    x_val_raw = df.loc[val_mask, features].fillna(0.0).to_numpy(dtype=np.float32)
    x_val_norm = np.nan_to_num((x_val_raw - feat_mean) / feat_std, nan=0.0)

    y_train_arr = df.loc[train_mask, "target_bias"].to_numpy(dtype=np.float32)
    y_val_arr = df.loc[val_mask, "target_bias"].to_numpy(dtype=np.float32)
    lats_train = lats_all[train_mask]
    lons_train = lons_all[train_mask]
    lats_val = lats_all[val_mask]
    lons_val = lons_all[val_mask]

    meta_cols = [c for c in [
        "valid_time", "run_time", "lead_time_hours", "lead_time", "weather_model",
        "station_id", "grid_id", "lat", "lon", "target_lat", "target_lon",
        "basin_id", "basin_name", "observed_rain", "nwp_rain_raw", "target_bias"
    ] if c in df.columns]
    df_val = df.loc[val_mask, meta_cols].copy()

    model = STGNNModel(in_dim=len(features), hidden_dim=16 if smoke_test else 32)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    n_train = len(x_train_norm)
    target_batch_nodes = 32 if smoke_test else min(512, max(64, n_train // 20))
    epochs = 2 if smoke_test else 15

    # Identify timestamp grouping column to preserve spatial snapshot integrity
    time_col = None
    for tc in ["valid_time", "run_time", "timestamp"]:
        if tc in df.columns:
            time_col = tc
            break

    # Build group indices for training by timestamp snapshot
    df_train_sub = df.loc[train_mask]
    if time_col is not None:
        grouped_dict = df_train_sub.groupby(time_col, sort=False).indices
        train_groups = [np.array(pos, dtype=np.int64) for pos in grouped_dict.values() if len(pos) > 0]
    else:
        train_groups = [np.arange(i, min(n_train, i + 64), dtype=np.int64) for i in range(0, n_train, 64)]

    # Bound block size to prevent excessive memory usage
    max_block_nodes = 256
    bounded_train_groups = []
    for g in train_groups:
        if len(g) > max_block_nodes:
            for s in range(0, len(g), max_block_nodes):
                bounded_train_groups.append(g[s:s + max_block_nodes])
        else:
            bounded_train_groups.append(g)
    train_groups = bounded_train_groups

    model.train()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(len(train_groups))
        shuffled_groups = [train_groups[i] for i in perm]

        batches = []
        cur_batch = []
        cur_count = 0
        for g in shuffled_groups:
            cur_batch.append(g)
            cur_count += len(g)
            if cur_count >= target_batch_nodes:
                batches.append(cur_batch)
                cur_batch = []
                cur_count = 0
        if cur_batch:
            batches.append(cur_batch)

        total_loss = 0.0
        for batch_groups in batches:
            batch_idx = np.concatenate(batch_groups)
            b_x = numpy_to_torch(x_train_norm[batch_idx])
            b_y = numpy_to_torch(y_train_arr[batch_idx])
            b_adj = build_batch_block_diagonal_adjacency(batch_groups, lats_train, lons_train)

            optimizer.zero_grad()
            out = model(b_x, b_adj)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(batches))
        logger.info("  -> [ST-GNN Epoch %d/%d] Training Huber Loss: %.4f", ep, epochs, avg_loss)

    # Save model checkpoint
    model_dir = output_dir / "models" / "stgnn" / weather_model / ablation
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_file)
    logger.info("Saved ST-GNN checkpoint: %s", ckpt_file)

    # Evaluate across validation set with timestamp-consistent spatial graph batching
    n_val = len(x_val_norm)
    df_val_sub = df.loc[val_mask]
    if time_col is not None:
        v_grouped_dict = df_val_sub.groupby(time_col, sort=False).indices
        val_groups = [np.array(pos, dtype=np.int64) for pos in v_grouped_dict.values() if len(pos) > 0]
    else:
        val_groups = [np.arange(i, min(n_val, i + 64), dtype=np.int64) for i in range(0, n_val, 64)]

    bounded_val_groups = []
    for g in val_groups:
        if len(g) > max_block_nodes:
            for s in range(0, len(g), max_block_nodes):
                bounded_val_groups.append(g[s:s + max_block_nodes])
        else:
            bounded_val_groups.append(g)
    val_groups = bounded_val_groups

    val_batches = []
    cur_v_batch = []
    cur_v_count = 0
    for g in val_groups:
        cur_v_batch.append(g)
        cur_v_count += len(g)
        if cur_v_count >= target_batch_nodes:
            val_batches.append(cur_v_batch)
            cur_v_batch = []
            cur_v_count = 0
    if cur_v_batch:
        val_batches.append(cur_v_batch)

    model.eval()
    val_pred_bias = np.zeros(n_val, dtype=np.float32)
    with torch.no_grad():
        for batch_groups in val_batches:
            batch_idx = np.concatenate(batch_groups)
            v_x = numpy_to_torch(x_val_norm[batch_idx])
            v_adj = build_batch_block_diagonal_adjacency(batch_groups, lats_val, lons_val)
            preds_batch = torch_to_numpy(model(v_x, v_adj)).flatten()
            val_pred_bias[batch_idx] = preds_batch

    nwp_raw_val = df_val["nwp_rain_raw"].to_numpy(dtype=float) if "nwp_rain_raw" in df_val.columns else np.zeros(len(df_val))
    obs_val = df_val["observed_rain"].to_numpy(dtype=float) if "observed_rain" in df_val.columns else (nwp_raw_val + y_val_arr)
    raw_rmse = float(np.sqrt(np.mean((nwp_raw_val - obs_val) ** 2)))
    corrected_val = np.maximum(0.0, nwp_raw_val + val_pred_bias)
    val_rmse = float(np.sqrt(np.mean((corrected_val - obs_val) ** 2)))
    diff_pct = ((raw_rmse - val_rmse) / raw_rmse) * 100.0 if raw_rmse > 0 else 0.0

    logger.info(
        "ST-GNN [%s] [%s] Val RMSE -> Before (Raw NWP): %.4f | After (Corrected): %.4f (Skill: %+.2f%%) [on %d val samples]",
        weather_model.upper(), ablation.upper(), raw_rmse, val_rmse, diff_pct, len(y_val_arr)
    )

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
    parser.add_argument("--weather-model", type=str, default=None, help="Target Weather Model (e.g. 'ecmwf_ifs', 'gfs', 'ecmwf_ifs,gfs', or 'all')")
    parser.add_argument("--ablation", type=str, default=None, help="Feature ablation variant (e.g. 'm1', 'm2', 'm3', 'm1,m2', or 'all')")
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
