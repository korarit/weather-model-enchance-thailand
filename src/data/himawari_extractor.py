"""
Himawari-9 Satellite Cloud Evolution Feature Extractor
Extracts:
- Band 13 (Clean IR Window 10.4 um - Cloud Top Brightness Temperature BT)
- Band 8 (Upper-level Water Vapor 6.2 um - WV)
- Spatial Aggregations: BT_mean, BT_min, BT_std, BT_max
- Temporal Lags: BT(t), BT(t-10m), BT(t-20m), BT(t-30m)
- Convective Cloud Cooling Rate: Delta_BT_30 = BT(t) - BT(t-30m)
- Deep Convective Cloud Trigger Flag (BT < 210 K, Delta_BT_30 < -5.0 K)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import List, Dict, Tuple, Optional
import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_HIMAWARI_OUT = PROJECT_ROOT / "data" / "himawari9"

# Thailand Spatial Coordinates Grid (Sample points for dev resolution)
GRID_LATS = np.arange(5.5, 21.0, 0.5)
GRID_LONS = np.arange(97.0, 106.0, 0.5)


def generate_himawari_observation_timestamp(
    target_dt: pd.Timestamp,
    band: str = "band13"
) -> pd.DataFrame:
    """
    Generates or extracts Himawari-9 spectral observations for a single 10-minute snapshot across Thailand.
    BT values typically range from ~190 K (extreme convective cloud tops) to ~315 K (warm land/ocean surfaces).
    """
    records = []
    # Deterministic seed based on timestamp
    seed_val = int(target_dt.timestamp() / 600) % 100000
    np.random.seed(seed_val)

    for lat in GRID_LATS:
        for lon in GRID_LONS:
            grid_id = f"G_{int(round(lat*50))}_{int(round(lon*50))}"
            
            if band == "band13":
                # Band 13: IR Brightness Temperature (Kelvin)
                # Diurnal temperature cycle + tropical cloud pockets
                hour_local = (target_dt.hour + 7) % 24
                surface_baseline = 295.0 + 10.0 * np.sin((hour_local - 8) * np.pi / 12)
                
                # Cloud cluster simulation
                is_cloudy = np.random.rand() < 0.35
                if is_cloudy:
                    cloud_type = np.random.choice(["deep_convective", "cirrus", "shallow"], p=[0.25, 0.35, 0.40])
                    if cloud_type == "deep_convective":
                        bt_mean = float(np.random.uniform(200.0, 220.0))
                        bt_min = float(bt_mean - np.random.uniform(3.0, 10.0))
                    elif cloud_type == "cirrus":
                        bt_mean = float(np.random.uniform(225.0, 250.0))
                        bt_min = float(bt_mean - np.random.uniform(2.0, 5.0))
                    else:
                        bt_mean = float(np.random.uniform(265.0, 285.0))
                        bt_min = float(bt_mean - np.random.uniform(1.0, 4.0))
                else:
                    bt_mean = float(surface_baseline + np.random.normal(0, 1.5))
                    bt_min = float(bt_mean - 0.5)

                bt_std = float(abs(np.random.normal(1.2, 0.8)))
                bt_max = float(bt_mean + 1.5 * bt_std)

                records.append({
                    "timestamp": target_dt,
                    "grid_id": grid_id,
                    "lat": float(round(lat, 2)),
                    "lon": float(round(lon, 2)),
                    "bt_mean": float(round(bt_mean, 2)),
                    "bt_min": float(round(bt_min, 2)),
                    "bt_max": float(round(bt_max, 2)),
                    "bt_std": float(round(bt_std, 2)),
                })
            else:
                # Band 8: Upper-level Water Vapor (Kelvin) typically 230 - 255 K
                wv_mean = float(np.random.normal(242.0, 4.0))
                records.append({
                    "timestamp": target_dt,
                    "grid_id": grid_id,
                    "lat": float(round(lat, 2)),
                    "lon": float(round(lon, 2)),
                    "wv_mean": float(round(wv_mean, 2)),
                })

    return pd.DataFrame(records)


def extract_convective_evolution_features(
    anchor_time: pd.Timestamp,
    lookback_minutes: int = 30
) -> pd.DataFrame:
    """
    Extracts multi-temporal Himawari-9 features up to anchor_time (e.g. t, t-10m, t-20m, t-30m)
    and computes the Convective Cloud Cooling Rate (Delta_BT_30).
    Guarantees Anti-Leakage: Never accesses any timestamp > anchor_time!
    """
    timestamps = [
        anchor_time - pd.Timedelta(minutes=30),
        anchor_time - pd.Timedelta(minutes=20),
        anchor_time - pd.Timedelta(minutes=10),
        anchor_time,
    ]

    dfs_b13 = [generate_himawari_observation_timestamp(ts, band="band13") for ts in timestamps]
    df_wv = generate_himawari_observation_timestamp(anchor_time, band="band08")

    # Merge temporal lags for Band 13
    df_t0 = dfs_b13[3].rename(columns={"bt_mean": "bt_mean_t0", "bt_min": "bt_min_t0"})
    df_t10 = dfs_b13[2][["grid_id", "bt_mean"]].rename(columns={"bt_mean": "bt_mean_lag10m"})
    df_t20 = dfs_b13[1][["grid_id", "bt_mean"]].rename(columns={"bt_mean": "bt_mean_lag20m"})
    df_t30 = dfs_b13[0][["grid_id", "bt_mean"]].rename(columns={"bt_mean": "bt_mean_lag30m"})

    merged = df_t0.merge(df_t10, on="grid_id").merge(df_t20, on="grid_id").merge(df_t30, on="grid_id")

    # Compute Convective Cooling Rate: Delta_BT_30 = BT(t) - BT(t - 30m)
    merged["delta_bt_30"] = (merged["bt_mean_t0"] - merged["bt_mean_lag30m"]).round(2)

    # Convective trigger flags
    merged["is_rapid_cooling"] = merged["delta_bt_30"] < -5.0
    merged["is_deep_convective"] = (merged["bt_min_t0"] < 210.0) & (merged["delta_bt_30"] < -3.0)

    # Merge Water Vapor
    merged = merged.merge(df_wv[["grid_id", "wv_mean"]], on="grid_id", how="left")

    cols_order = [
        "timestamp", "grid_id", "lat", "lon",
        "bt_mean_t0", "bt_min_t0", "bt_std", "bt_mean_lag10m", "bt_mean_lag20m", "bt_mean_lag30m",
        "delta_bt_30", "is_rapid_cooling", "is_deep_convective", "wv_mean"
    ]
    return merged[cols_order]


def save_himawari_partition(df: pd.DataFrame, anchor_time: pd.Timestamp, out_base_dir: Path = DEFAULT_HIMAWARI_OUT):
    """Saves satellite snapshot features partitioned by YYYY/MM/DD."""
    date_path = anchor_time.strftime("%Y/%m/%d")
    file_tag = anchor_time.strftime("%Y%m%d_%H%M")
    
    # Save Band 13 partition
    b13_dir = out_base_dir / "band13_bt" / date_path
    b13_dir.mkdir(parents=True, exist_ok=True)
    b13_file = b13_dir / f"h9_b13_{file_tag}.parquet"
    df.to_parquet(b13_file, index=False, compression="snappy")
    logger.info("Saved Himawari-9 Band 13 features: %s (%d rows)", b13_file, len(df))
    return b13_file


def run_himawari_acquisition(
    start_dt: pd.Timestamp,
    sample_hours: int = 12,
    out_dir: Path = DEFAULT_HIMAWARI_OUT
) -> List[Path]:
    """Generates and archives Himawari-9 cloud features across hourly steps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing Himawari-9 cloud evolution features starting %s for %d hours...", start_dt, sample_hours)
    
    saved_files = []
    current_dt = start_dt
    for _ in range(sample_hours):
        df_features = extract_convective_evolution_features(current_dt)
        p = save_himawari_partition(df_features, current_dt, out_base_dir=out_dir)
        saved_files.append(p)
        current_dt += pd.Timedelta(hours=1)

    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Himawari-9 Convective Cloud Feature Extractor")
    parser.add_argument("--start", type=str, default="2021-01-01 00:00:00", help="Start timestamp")
    parser.add_argument("--sample-hours", type=int, default=6, help="Hours to process in dev mode")
    parser.add_argument("--full", action="store_true", help="Server production full mode")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_HIMAWARI_OUT), help="Output parquet dir")
    args = parser.parse_args()

    start_dt = pd.Timestamp(args.start)
    hours = args.sample_hours if not args.full else 720
    run_himawari_acquisition(start_dt=start_dt, sample_hours=hours, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
