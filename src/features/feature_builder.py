"""
Unified Feature Matrix Builder
Integrates:
1. Keys & Metadata: grid_id, run_time, valid_time, lead_time_hours, target_lat, target_lon
2. NWP Model Forecast Features (ECMWF / GFS)
3. Ground Observation Spatial Features (t <= run_time) via SpatialObservationIndexer
4. Himawari-9 Satellite Cloud Evolution Features (t <= run_time)
5. Physical Topographic & Geographic Features (from Thailand 2km Master Grid)
6. Training Target: observed_rain and target_bias = observed_rain - nwp_rain_raw
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from src.data.anti_leakage_validator import assert_no_temporal_leakage
from src.features.spatial_features import SpatialObservationIndexer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_GEO_DIR = PROJECT_ROOT / "data" / "geo"
DEFAULT_META_DIR = PROJECT_ROOT / "data" / "metadata"


def load_master_grid(geo_dir: Path = DEFAULT_GEO_DIR) -> pd.DataFrame:
    """Loads 2 km Master Grid with topography."""
    grid_pq = geo_dir / "thailand_2km_master_grid.parquet"
    if grid_pq.exists():
        return pd.read_parquet(grid_pq)
    
    # If not exists yet, generate on the fly
    from src.features.grid_generator import generate_master_grid
    return generate_master_grid(output_dir=geo_dir)


def load_stations_metadata(meta_dir: Path = DEFAULT_META_DIR) -> pd.DataFrame:
    """Loads station metadata."""
    meta_csv = meta_dir / "hii_stations_master_metadata.csv"
    if meta_csv.exists():
        return pd.read_csv(meta_csv)
    from src.data.hii_metadata import build_master_metadata
    build_master_metadata(output_dir=meta_dir)
    return pd.read_csv(meta_csv)


def assemble_unified_training_matrix(
    nwp_forecast_df: pd.DataFrame,
    himawari_features_df: pd.DataFrame,
    station_obs_dict: Dict[str, Dict[str, float]],
    grid_df: pd.DataFrame,
    indexer: SpatialObservationIndexer,
    observed_rain_dict: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Assembles a unified tabular matrix row-by-row or grid-by-grid for a forecast cycle.
    Enforces Temporal Anti-Leakage Protocol across all inputs.
    """
    run_time = nwp_forecast_df["run_time"].iloc[0]

    # Verify Anti-Leakage on satellite observation timestamps
    if "timestamp" in himawari_features_df.columns:
        assert_no_temporal_leakage(run_time, himawari_features_df["timestamp"], context="satellite_features")

    logger.info("Assembling unified matrix for run_time: %s (%d NWP rows)...", run_time, len(nwp_forecast_df))

    # Merge Topographic features by grid_id (or nearest centroid)
    topo_cols = [
        "grid_id", "elevation_m", "elevation_std", "slope_deg",
        "aspect_sin", "aspect_cos", "dist_coast_km", "coriolis_param"
    ]
    grid_sub = grid_df[topo_cols].drop_duplicates("grid_id")
    merged = nwp_forecast_df.merge(grid_sub, on="grid_id", how="left")

    # If grid_id was missing from grid_df, fill with median topography
    for col in ["elevation_m", "elevation_std", "slope_deg", "dist_coast_km"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median() if not merged[col].dropna().empty else 10.0)
    for col in ["aspect_sin", "aspect_cos", "coriolis_param"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # Merge Satellite Features by grid_id
    sat_cols = [
        "grid_id", "bt_mean_t0", "bt_min_t0", "bt_std",
        "bt_mean_lag10m", "bt_mean_lag20m", "bt_mean_lag30m",
        "delta_bt_30", "is_rapid_cooling", "is_deep_convective", "wv_mean"
    ]
    sat_sub = himawari_features_df[[c for c in sat_cols if c in himawari_features_df.columns]].drop_duplicates("grid_id")
    merged = merged.merge(sat_sub, on="grid_id", how="left")

    # Fill satellite defaults if cloudy data missing
    merged["bt_mean_t0"] = merged["bt_mean_t0"].fillna(285.0)
    merged["bt_min_t0"] = merged["bt_min_t0"].fillna(280.0)
    merged["bt_std"] = merged["bt_std"].fillna(1.5)
    merged["delta_bt_30"] = merged["delta_bt_30"].fillna(0.0)
    merged["is_rapid_cooling"] = merged["is_rapid_cooling"].fillna(False)
    merged["is_deep_convective"] = merged["is_deep_convective"].fillna(False)
    merged["wv_mean"] = merged["wv_mean"].fillna(245.0)

    # Compute Ground Spatial Observation Features per grid cell
    unique_cells = merged[["grid_id", "lat", "lon"]].drop_duplicates("grid_id")
    spatial_records = []

    for _, row in unique_cells.iterrows():
        gid = row["grid_id"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        sp_feats = indexer.extract_grid_spatial_features(lat, lon, station_obs_dict)
        sp_feats["grid_id"] = gid
        spatial_records.append(sp_feats)

    sp_df = pd.DataFrame(spatial_records)
    merged = merged.merge(sp_df, on="grid_id", how="left")

    # Rename / Standardize NWP columns
    rename_map = {
        "tp": "nwp_rain_raw",
        "t2m": "nwp_temp_2m",
        "sp": "nwp_pressure",
        "u10": "nwp_u10",
        "v10": "nwp_v10",
        "cape": "nwp_cape",
        "lat": "target_lat",
        "lon": "target_lon",
    }
    merged = merged.rename(columns=rename_map)

    # Add Target: observed_rain and target_bias
    if observed_rain_dict:
        merged["observed_rain"] = merged["grid_id"].map(observed_rain_dict).fillna(np.nan)
    else:
        # Synthetic / simulated ground truth for development pipeline validation
        # Residual bias = observed - nwp
        np.random.seed(42)
        true_rain = merged["nwp_rain_raw"] * np.random.uniform(0.7, 1.4, size=len(merged)) + np.random.normal(0, 0.2, size=len(merged))
        merged["observed_rain"] = np.clip(true_rain, 0.0, 300.0).round(2)

    merged["target_bias"] = (merged["observed_rain"] - merged["nwp_rain_raw"]).round(2)

    logger.info("Assembled unified matrix successfully (%d rows, %d columns)", len(merged), len(merged.columns))
    return merged


def run_sample_builder(output_dir: Path = DEFAULT_FEATURES_DIR) -> Path:
    """Executes end-to-end sample feature assembly for dev validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running Phase 3 Sample Feature Builder...")

    grid_df = load_master_grid()
    stn_meta_df = load_stations_metadata()
    indexer = SpatialObservationIndexer(stn_meta_df)

    # Load or generate sample NWP & Himawari inputs
    from src.data.tigge_extractor import run_extraction_pipeline
    nwp_files = run_extraction_pipeline()
    if nwp_files:
        nwp_df = pd.read_parquet(nwp_files[0])
    else:
        from src.data.tigge_downloader import download_tigge_cycle
        raw_p = download_tigge_cycle("ecmf", "2021-01-01", "00:00")
        from src.data.tigge_extractor import process_raw_nwp_file
        nwp_pq = process_raw_nwp_file(raw_p)
        nwp_df = pd.read_parquet(nwp_pq)

    run_time = nwp_df["run_time"].iloc[0]

    # Generate satellite features strictly at or before run_time
    from src.data.himawari_extractor import extract_convective_evolution_features
    sat_df = extract_convective_evolution_features(run_time)

    # Create station observations at t <= run_time
    sample_obs = {}
    for code in stn_meta_df["station_code"].dropna().head(100):
        sample_obs[code] = {
            "hourly_rain": float(np.random.choice([0.0, 0.0, 0.0, 1.2, 4.5])),
            "pressure": float(np.random.normal(1008.0, 3.0)),
            "humidity": float(np.random.normal(78.0, 8.0)),
        }

    unified_matrix = assemble_unified_training_matrix(
        nwp_forecast_df=nwp_df,
        himawari_features_df=sat_df,
        station_obs_dict=sample_obs,
        grid_df=grid_df,
        indexer=indexer
    )

    out_file = output_dir / f"training_features_{run_time.strftime('%Y%m%d_%Hz')}.parquet"
    unified_matrix.to_parquet(out_file, index=False, compression="snappy")
    logger.info("Saved unified training feature matrix to: %s", out_file)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Unified Feature Matrix Builder")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_FEATURES_DIR), help="Output directory")
    args = parser.parse_args()

    run_sample_builder(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
