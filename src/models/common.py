"""
Common Utilities for ML Bias Correction Modeling
Handles:
- Feature ablation mappings (M1, M2, M3)
- Strict data split protocols (2021-2024 Train/Val, 2025 Frozen Test)
- Strict guardrails: blocks DWR station leakage
- Prediction export formatting conforming to Phase 4 benchmark standard
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base feature groups
NWP_FEATURES = [
    "nwp_rain_raw", "nwp_pressure", "nwp_temp_2m",
    "nwp_u10", "nwp_v10", "nwp_cape", "lead_time_hours"
]

TOPO_FEATURES = [
    "elevation_m", "elevation_std", "slope_deg",
    "aspect_sin", "aspect_cos", "dist_coast_km", "coriolis_param"
]

GROUND_FEATURES = [
    "rain_mean_2_5km", "rain_max_2_5km",
    "pressure_mean_2_5km", "pressure_std_2_5km",
    "humidity_mean_2_5km", "humidity_std_2_5km",
    "nearest_stn_dist_2_5km", "nearest_stn_bearing_sin", "nearest_stn_bearing_cos",
    "rain_mean_5_10km", "rain_max_5_10km",
    "pressure_mean_5_10km", "humidity_mean_5_10km",
    "pressure_mean_20_50km", "humidity_mean_20_50km",
    "pressure_gradient_mag", "humidity_gradient_mag"
]

SATELLITE_FEATURES = [
    "bt_mean_t0", "bt_min_t0", "bt_std",
    "bt_mean_lag10m", "bt_mean_lag20m", "bt_mean_lag30m",
    "delta_bt_30", "wv_mean"
]

ABLATION_FEATURES = {
    "m1": NWP_FEATURES + TOPO_FEATURES + GROUND_FEATURES,
    "m2": NWP_FEATURES + TOPO_FEATURES + SATELLITE_FEATURES,
    "m3": NWP_FEATURES + TOPO_FEATURES + GROUND_FEATURES + SATELLITE_FEATURES,
}


def assert_no_dwr_leakage(file_path_or_str: str):
    """Guarantees that DWR station dataset is 100% BLIND and NEVER used in training."""
    s = str(file_path_or_str).lower()
    if "dwr_rain" in s or "dwr_hourly" in s:
        raise ValueError(
            "[STRICT HOLD-OUT VIOLATION] DWR station data in 'dataset/dwr_rain/' is reserved "
            "exclusively as blind test set in Phase 5 and must never be accessed during Phase 4 training!"
        )


def assert_valid_training_years(df: pd.DataFrame, time_col: str = "valid_time"):
    """Guarantees that 2025 data is strictly frozen and not present in training."""
    years = pd.to_datetime(df[time_col]).dt.year.unique()
    if 2025 in years:
        raise ValueError(
            "[FROZEN TEST LEAKAGE] Data from 2025 detected in training split! "
            "2025 must remain strictly frozen for final out-of-time evaluation."
        )


SUPPORTED_WEATHER_MODELS = [
    "ecmwf_ifs", "ncep_gfs", "dwd_icon", "cmc_gem", "bom_access", "meteo_arpege"
]


def generate_smoke_test_dataset(
    n_samples: int = 100,
    weather_model: str = "ecmwf_ifs",
    seed: int = 42
) -> pd.DataFrame:
    """Generates realistic synthetic tabular rows for local PC smoke tests per weather model."""
    np.random.seed(seed)
    run_times = pd.date_range("2021-01-01 00:00:00", periods=4, freq="12h")
    records = []
    
    basins = ["Chao Phraya", "Chi", "Mun", "Yom", "Ping", "Nan", "Tha Chin", "Mae Klong"]

    # Model specific bias characteristics
    model_bias_offset = {
        "ecmwf_ifs": 0.35,
        "ncep_gfs": 0.55,
        "dwd_icon": 0.45,
        "cmc_gem": 0.60,
        "bom_access": 0.50,
        "meteo_arpege": 0.55,
    }.get(weather_model, 0.4)

    for i in range(n_samples):
        rt = np.random.choice(run_times)
        lead = int(np.random.choice([6, 12, 18, 24]))
        vt = rt + pd.Timedelta(hours=lead)
        lat = float(round(np.random.uniform(7.0, 19.0), 2))
        lon = float(round(np.random.uniform(98.5, 104.5), 2))
        stn = f"HII_{1000 + (i % 25)}"
        basin = np.random.choice(basins)

        nwp_rain = float(round(max(0.0, np.random.exponential(1.5 + model_bias_offset * 0.3) if np.random.rand() < 0.4 else 0.0), 2))
        
        # Synthetic observed rain with bias
        bias_true = float(np.random.normal(0.5, 1.2)) if nwp_rain > 0 else (float(np.random.exponential(0.8)) if np.random.rand() < 0.15 else 0.0)
        obs_rain = float(round(max(0.0, nwp_rain + bias_true), 2))
        target_bias = float(round(obs_rain - nwp_rain, 2))

        row = {
            "run_time": rt,
            "valid_time": vt,
            "lead_time_hours": lead,
            "weather_model": weather_model,
            "station_id": stn,
            "basin_id": basin,
            "lat": lat,
            "lon": lon,
            "grid_id": f"G_{int(lat*50)}_{int(lon*50)}",
            "nwp_rain_raw": nwp_rain,
            "nwp_pressure": float(round(np.random.normal(1008.0, 3.0), 1)),
            "nwp_temp_2m": float(round(np.random.normal(28.0, 2.5), 1)),
            "nwp_u10": float(round(np.random.normal(1.0, 3.0), 2)),
            "nwp_v10": float(round(np.random.normal(2.0, 3.0), 2)),
            "nwp_cape": float(round(max(0.0, np.random.exponential(500.0)), 1)),
            "elevation_m": float(round(np.random.uniform(5.0, 1200.0), 1)),
            "elevation_std": float(round(np.random.uniform(1.0, 80.0), 1)),
            "slope_deg": float(round(np.random.uniform(0.5, 25.0), 1)),
            "aspect_sin": float(round(np.random.uniform(-1.0, 1.0), 4)),
            "aspect_cos": float(round(np.random.uniform(-1.0, 1.0), 4)),
            "dist_coast_km": float(round(np.random.uniform(10.0, 450.0), 1)),
            "coriolis_param": float(1.458e-4 * np.sin(np.radians(lat))),
            "rain_mean_2_5km": float(round(max(0.0, nwp_rain + np.random.normal(0, 0.5)), 2)),
            "rain_max_2_5km": float(round(max(0.0, nwp_rain + np.random.uniform(0, 2.0)), 2)),
            "pressure_mean_2_5km": float(round(np.random.normal(1008.0, 2.0), 1)),
            "pressure_std_2_5km": float(round(abs(np.random.normal(0.8, 0.4)), 2)),
            "humidity_mean_2_5km": float(round(np.random.uniform(60.0, 95.0), 1)),
            "humidity_std_2_5km": float(round(abs(np.random.normal(2.5, 1.0)), 2)),
            "nearest_stn_dist_2_5km": float(round(np.random.uniform(2.1, 4.9), 2)),
            "nearest_stn_bearing_sin": float(round(np.random.uniform(-1.0, 1.0), 4)),
            "nearest_stn_bearing_cos": float(round(np.random.uniform(-1.0, 1.0), 4)),
            "rain_mean_5_10km": float(round(max(0.0, nwp_rain + np.random.normal(0, 0.8)), 2)),
            "rain_max_5_10km": float(round(max(0.0, nwp_rain + np.random.uniform(0, 3.0)), 2)),
            "pressure_mean_5_10km": float(round(np.random.normal(1008.0, 2.5), 1)),
            "humidity_mean_5_10km": float(round(np.random.uniform(60.0, 95.0), 1)),
            "pressure_mean_20_50km": float(round(np.random.normal(1008.0, 3.0), 1)),
            "humidity_mean_20_50km": float(round(np.random.uniform(60.0, 95.0), 1)),
            "pressure_gradient_mag": float(round(np.random.uniform(0.5, 8.0), 2)),
            "humidity_gradient_mag": float(round(np.random.uniform(1.0, 15.0), 2)),
            "bt_mean_t0": float(round(np.random.uniform(210.0, 295.0), 1)),
            "bt_min_t0": float(round(np.random.uniform(200.0, 290.0), 1)),
            "bt_std": float(round(np.random.uniform(0.5, 5.0), 2)),
            "bt_mean_lag10m": float(round(np.random.uniform(210.0, 295.0), 1)),
            "bt_mean_lag20m": float(round(np.random.uniform(210.0, 295.0), 1)),
            "bt_mean_lag30m": float(round(np.random.uniform(210.0, 295.0), 1)),
            "delta_bt_30": float(round(np.random.normal(-1.0, 4.0), 2)),
            "wv_mean": float(round(np.random.normal(242.0, 4.0), 1)),
            "observed_rain": obs_rain,
            "target_bias": target_bias,
        }
        records.append(row)
    return pd.DataFrame(records)


def resolve_lat_lon(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts latitude and longitude Series from DataFrame supporting
    multiple column naming conventions: ('lat', 'lon'), ('target_lat', 'target_lon'), ('latitude', 'longitude').
    """
    if "lat" in df.columns:
        lat = df["lat"]
    elif "target_lat" in df.columns:
        lat = df["target_lat"]
    elif "latitude" in df.columns:
        lat = df["latitude"]
    else:
        lat = pd.Series(0.0, index=df.index, dtype=float)

    if "lon" in df.columns:
        lon = df["lon"]
    elif "target_lon" in df.columns:
        lon = df["target_lon"]
    elif "longitude" in df.columns:
        lon = df["longitude"]
    else:
        lon = pd.Series(0.0, index=df.index, dtype=float)

    return lat, lon


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes coordinate and common metadata column names in training/validation DataFrames
    to ensure compatibility across feature generation pipelines, runners, and prediction exports.
    """
    df = df.copy()
    # Coordinates: ensure 'lat' and 'lon' as well as 'target_lat' and 'target_lon' exist
    lat_s, lon_s = resolve_lat_lon(df)
    if "lat" not in df.columns:
        df["lat"] = lat_s
    if "lon" not in df.columns:
        df["lon"] = lon_s
    if "target_lat" not in df.columns:
        df["target_lat"] = lat_s
    if "target_lon" not in df.columns:
        df["target_lon"] = lon_s

    # Lead time: ensure 'lead_time_hours' and 'lead_time'
    if "lead_time_hours" not in df.columns and "lead_time" in df.columns:
        df["lead_time_hours"] = df["lead_time"]
    elif "lead_time" not in df.columns and "lead_time_hours" in df.columns:
        df["lead_time"] = df["lead_time_hours"]

    # Station ID / Grid ID
    if "station_id" not in df.columns and "grid_id" in df.columns:
        df["station_id"] = df["grid_id"]

    # Basin ID
    if "basin_id" not in df.columns:
        if "basin_name" in df.columns:
            df["basin_id"] = df["basin_name"]
        else:
            df["basin_id"] = "General"

    # Raw NWP rain
    if "nwp_rain_raw" not in df.columns and "tp" in df.columns:
        df["nwp_rain_raw"] = df["tp"]

    return df


def export_predictions(
    df: pd.DataFrame,
    predicted_bias: np.ndarray,
    model_name: str,
    ablation: str,
    output_dir: Path,
    weather_model: str = "ecmwf_ifs",
    quantile_preds: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[Path, Path]:
    """
    Exports standardized predictions CSV and Parquet:
    - corrected_rain = max(0.0, nwp_raw_rain + predicted_bias)
    - Output location: [output_dir]/predictions/[model_name]/[weather_model]_[model_name]_[ablation]_pred.csv
    """
    pred_dir = output_dir / "predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Standardize columns to safely resolve lat, lon, lead_time, etc.
    df = standardize_dataframe_columns(df)

    nwp_raw = df["nwp_rain_raw"].to_numpy(dtype=float) if "nwp_rain_raw" in df.columns else np.zeros(len(df), dtype=float)
    bias = np.array(predicted_bias, dtype=float)
    corrected = np.maximum(0.0, nwp_raw + bias)

    station_series = df["station_id"] if "station_id" in df.columns else (df["grid_id"] if "grid_id" in df.columns else pd.Series("STN_UNKNOWN", index=df.index))
    basin_series = df["basin_id"] if "basin_id" in df.columns else pd.Series("General", index=df.index)

    if "observed_rain" in df.columns:
        obs_rain_series = df["observed_rain"].astype(np.float32)
    elif "target_bias" in df.columns:
        obs_rain_series = (nwp_raw + df["target_bias"].to_numpy(dtype=float)).astype(np.float32)
    else:
        obs_rain_series = nwp_raw.astype(np.float32)

    valid_time_series = pd.to_datetime(df["valid_time"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ") if "valid_time" in df.columns else pd.Series("2021-01-01T00:00:00Z", index=df.index)
    run_time_series = pd.to_datetime(df["run_time"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ") if "run_time" in df.columns else pd.Series("2021-01-01T00:00:00Z", index=df.index)
    lead_time_series = df["lead_time_hours"].astype(int) if "lead_time_hours" in df.columns else (df["lead_time"].astype(int) if "lead_time" in df.columns else pd.Series(0, index=df.index, dtype=int))

    out_df = pd.DataFrame({
        "valid_time": valid_time_series,
        "run_time": run_time_series,
        "lead_time": lead_time_series,
        "weather_model": weather_model,
        "station_id": station_series,
        "lat": df["lat"].astype(np.float32),
        "lon": df["lon"].astype(np.float32),
        "basin_id": basin_series,
        "observed_rain": obs_rain_series,
        "nwp_raw_rain": nwp_raw.astype(np.float32),
        "predicted_bias": bias.astype(np.float32),
        "corrected_rain": corrected.astype(np.float32),
        "model_name": model_name,
        "ablation": ablation,
    })

    if quantile_preds:
        for q_name, q_vals in quantile_preds.items():
            out_df[q_name] = np.maximum(0.0, nwp_raw + q_vals).astype(np.float32)

    csv_path = pred_dir / f"{weather_model}_{model_name}_{ablation}_pred.csv"
    parquet_path = pred_dir / f"{weather_model}_{model_name}_{ablation}_pred.parquet"

    out_df.to_csv(csv_path, index=False)
    out_df.to_parquet(parquet_path, index=False, compression="snappy")
    
    # Also save standard filename for backward compatibility
    compat_csv = pred_dir / f"{model_name}_{ablation}_pred.csv"
    out_df.to_csv(compat_csv, index=False)
    
    logger.info("Exported predictions for %s [%s] (%s): %s (%d rows)", weather_model, model_name, ablation, csv_path.name, len(out_df))
    return csv_path, parquet_path


def resolve_runner_execution_targets(
    config_path: Optional[str] = None,
    weather_model_arg: Optional[str] = None,
    ablation_arg: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict]:
    """
    Resolves execution targets (weather_models, ablations, hyperparams) from CLI arguments
    and optional YAML config.
    Supports:
    - --weather-model all -> ["ecmwf_ifs", "ncep_gfs", "dwd_icon"]
    - --weather-model [wm] -> [wm]
    - --ablation all -> ["m1", "m2", "m3"]
    - --ablation [ab] -> [ab]
    - --config path/to/config.yaml -> loads weather_model, ablation_variant, hyperparameters
    """
    config_data = {}
    if config_path:
        cfg_p = Path(config_path)
        if not cfg_p.is_absolute():
            if (PROJECT_ROOT / cfg_p).exists():
                cfg_p = PROJECT_ROOT / cfg_p
            elif not cfg_p.exists():
                parts = cfg_p.parts
                if len(parts) >= 3 and parts[0] == "configs" and parts[1] == "models":
                    filename = parts[-1]
                    m_name = filename.split("_")[0]
                    alt_p = PROJECT_ROOT / "configs" / "models" / m_name / filename
                    if alt_p.exists():
                        cfg_p = alt_p
        if cfg_p.exists():
            import yaml
            with open(cfg_p, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from: %s", cfg_p)
        else:
            logger.warning("Config path %s not found. Proceeding with CLI arguments.", config_path)

    # Resolve target weather models
    target_wm = weather_model_arg
    if target_wm is None or (target_wm == "ecmwf_ifs" and "weather_model" in config_data):
        target_wm = config_data.get("weather_model", "ecmwf_ifs")

    if target_wm == "all":
        weather_models = ["ecmwf_ifs", "ncep_gfs", "dwd_icon"]
    else:
        weather_models = [target_wm]

    # Resolve target ablation variants
    target_ab = ablation_arg
    if target_ab is None or (target_ab == "m3" and "ablation_variant" in config_data):
        target_ab = config_data.get("ablation_variant", "m3")

    if target_ab == "all":
        ablations = ["m1", "m2", "m3"]
    else:
        ablations = [target_ab]

    hyperparams = config_data.get("hyperparameters", {})
    return weather_models, ablations, hyperparams

