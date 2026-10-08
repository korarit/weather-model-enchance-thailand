"""
Hybrid NWP Downloader & Transformer:
Combines real operational archives from:
  1. NOAA GFS: Directly from AWS S3 ('s3://noaa-gfs-bdp-pds/') via unsigned byte-range GRIB2 queries
     - Real model run cycles (00z, 06z, 12z, 18z) and genuine lead times (f001 to f024)
     - Covers 2021-2025+ complete archive
  2. ECMWF IFS: From Open-Meteo Historical Forecast API
     - Seamless hourly series transformed with a Rolling-Window lead time mapping (1 to 24h)

Produces identical Analysis-Ready Parquet files matching the system schema:
  ['origin', 'run_time', 'valid_time', 'lead_time_hours', 'grid_id', 'lat', 'lon',
   'tp', 'sp', 't2m', 'u10', 'v10', 'cape']
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import rasterio
from rasterio.io import MemoryFile

from src.config.paths import (
    DEFAULT_OUT_NWP_DIR,
    DEFAULT_RAW_NWP_DIR,
    DEFAULT_HII_META_DIR,
    resolve_forecast_paths,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Endpoints & Buckets
OPEN_METEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
NOAA_GFS_S3_BUCKET = "noaa-gfs-bdp-pds"

# Strictly support ONLY these 2 models
SUPPORTED_HYBRID_MODELS = {
    "ecmwf_ifs": {
        "primary_source": "openmeteo",
        "open_meteo_model": "ecmwf_ifs",
        "origin_name": "ecmwf_ifs",
        "description": "ECMWF IFS (via Open-Meteo + Rolling Window)",
    },
    "gfs": {
        "primary_source": "aws",
        "open_meteo_model": "gfs_seamless",
        "origin_name": "gfs",
        "description": "NOAA NCEP GFS (Direct from AWS S3 noaa-gfs-bdp-pds)",
    },
}

MODEL_ALIASES = {
    "ecmwf_ifs": "ecmwf_ifs",
    "ecmwf": "ecmwf_ifs",
    "ecmf": "ecmwf_ifs",
    "gfs": "gfs",
    "gfs_seamless": "gfs",
    "gfs_global": "gfs",
    "ncep": "gfs",
    "kwbc": "gfs",
}

DEFAULT_CYCLES = ["00:00", "12:00"]
DEFAULT_MAX_LEAD = 24

# Default Thailand sample coordinates if station metadata is unavailable
DEFAULT_SAMPLE_LOCATIONS = [
    (13.75, 100.50),  # Bangkok (Central)
    (18.78, 98.98),   # Chiang Mai (North)
    (16.43, 102.83),  # Khon Kaen (Northeast)
    (7.00, 100.47),   # Songkhla / Hat Yai (South)
    (12.68, 101.28),  # Rayong (East)
    (14.00, 99.50),   # Kanchanaburi (West)
]


def resolve_model_key(model: str) -> str:
    """Validates and resolves requested model name to canonical key ('ecmwf_ifs' or 'gfs')."""
    m_clean = model.strip().lower()
    if m_clean in MODEL_ALIASES:
        return MODEL_ALIASES[m_clean]
    raise ValueError(
        f"Model '{model}' is not supported by hybrid downloader. "
        f"Hybrid downloader strictly supports only 2 models: 'ecmwf_ifs' and 'gfs' "
        f"(Accepted aliases: {list(MODEL_ALIASES.keys())})"
    )


def compute_uv_components(speed: Union[float, np.ndarray], direction_deg: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes meteorological U (eastward) and V (northward) wind components
    from wind speed (m/s) and wind direction (degrees).
    """
    rad = np.radians(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def generate_grid_id(lat: float, lon: float) -> str:
    """Generates standard 2 km grid ID consistent with tigge_extractor and feature_builder."""
    return f"G_{round(lat * 50)}_{round(lon * 50)}"


def load_locations(
    meta_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    custom_coords: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Loads station coordinates from metadata or falls back to sample locations."""
    if custom_coords:
        return custom_coords

    meta_dir = meta_dir or DEFAULT_HII_META_DIR
    meta_csv = meta_dir / "hii_stations_master_metadata.csv"

    if meta_csv.exists():
        try:
            df = pd.read_csv(meta_csv)
            if "latitude" in df.columns and "longitude" in df.columns:
                valid = df[["latitude", "longitude"]].dropna().drop_duplicates()
                coords = [(float(r["latitude"]), float(r["longitude"])) for _, r in valid.iterrows()]
                if limit and limit > 0:
                    coords = coords[:limit]
                if coords:
                    logger.info("Loaded %d station coordinates from %s", len(coords), meta_csv.name)
                    return coords
        except Exception as e:
            logger.warning("Could not read coordinates from %s: %s", meta_csv, e)

    logger.info("Using %d default Thailand sample coordinates.", len(DEFAULT_SAMPLE_LOCATIONS))
    return DEFAULT_SAMPLE_LOCATIONS if not limit else DEFAULT_SAMPLE_LOCATIONS[:limit]


# =============================================================================
# PART 1: NOAA GFS DOWNLOADER FROM AWS S3 (noaa-gfs-bdp-pds)
# =============================================================================

def get_s3_client():
    """Returns anonymous boto3 client for public NOAA S3 access."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def resolve_gfs_s3_key(date_str: str, cycle_hour: int, lead_step: int, s3_client=None) -> Optional[str]:
    """
    Resolves S3 key for GFS GRIB2 file.
    Supports both GFSv16 (atmos/ path from late March 2021 to present)
    and GFSv15 (root cycle path in early 2021).
    """
    s3 = s3_client or get_s3_client()
    d_clean = date_str.replace("-", "")
    hh = f"{cycle_hour:02d}"
    fxx = f"f{lead_step:03d}"

    # Priority 1: GFSv16+ (late March 2021 onwards)
    k1 = f"gfs.{d_clean}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.{fxx}"
    # Priority 2: GFSv15 (early 2021)
    k2 = f"gfs.{d_clean}/{hh}/gfs.t{hh}z.pgrb2.0p25.{fxx}"

    for candidate in [k1, k2]:
        try:
            s3.head_object(Bucket=NOAA_GFS_S3_BUCKET, Key=f"{candidate}.idx")
            return candidate
        except Exception:
            pass
    return None


def fetch_gfs_s3_single_lead(
    key: str,
    coords: List[Tuple[float, float]],
    s3_client=None,
) -> Optional[Dict[str, List[float]]]:
    """
    Downloads byte-ranges of required weather variables from AWS S3 using the .idx file,
    and samples values at specified coordinate locations using in-memory rasterio.
    """
    s3 = s3_client or get_s3_client()
    idx_key = f"{key}.idx"

    try:
        idx_body = s3.get_object(Bucket=NOAA_GFS_S3_BUCKET, Key=idx_key)["Body"].read().decode("utf-8")
        idx_lines = idx_body.splitlines()
    except Exception as e:
        logger.warning("Could not fetch .idx for %s: %s", key, e)
        return None

    def get_var_byte_range(pattern: str) -> Tuple[Optional[int], Optional[int]]:
        for i, line in enumerate(idx_lines):
            if pattern in line:
                start = int(line.split(":")[1])
                end = int(idx_lines[i + 1].split(":")[1]) - 1 if i + 1 < len(idx_lines) else None
                return start, end
        return None, None

    vars_needed = {
        "t2m": "TMP:2 m above ground",
        "sp": "PRES:surface",
        "u10": "UGRD:10 m above ground",
        "v10": "VGRD:10 m above ground",
        "apcp": "APCP:surface",
        "cape": "CAPE:surface",
    }

    sample_pts = [(lon, lat) for lat, lon in coords]
    results = {}

    for var_name, pattern in vars_needed.items():
        start, end = get_var_byte_range(pattern)
        if start is None:
            results[var_name] = [0.0] * len(coords)
            continue

        rng = f"bytes={start}-{end}" if end else f"bytes={start}-"
        try:
            part = s3.get_object(Bucket=NOAA_GFS_S3_BUCKET, Key=key, Range=rng)["Body"].read()
            with MemoryFile(part) as memfile:
                with memfile.open() as src:
                    vals = [float(v[0]) for v in src.sample(sample_pts)]
                    results[var_name] = vals
        except Exception as e:
            logger.warning("Error reading %s from %s: %s", var_name, key, e)
            results[var_name] = [0.0] * len(coords)

    # Unit conversions:
    # sp: Pascals -> hPa
    results["sp"] = [p / 100.0 if p > 2000.0 else p for p in results.get("sp", [])]
    return results


def download_gfs_cycle_from_aws(
    date_str: str,
    cycle_str: str,
    coords: List[Tuple[float, float]],
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Downloads full forecast horizon of a GFS cycle from NOAA AWS S3
    and formats into analysis-ready records with genuine lead_time_hours.
    """
    s3 = get_s3_client()
    chour = int(cycle_str.split(":")[0])
    run_time = pd.Timestamp(f"{date_str} {cycle_str}")

    logger.info("Ingesting GFS directly from AWS S3 (noaa-gfs-bdp-pds) for %s %sz across %d location(s)...",
                date_str, f"{chour:02d}", len(coords))

    # Resolve S3 keys for available lead steps
    lead_keys = {}
    for lead in range(1, max_lead_hours + 1):
        k = resolve_gfs_s3_key(date_str, chour, lead, s3_client=s3)
        if k:
            lead_keys[lead] = k

    if not lead_keys:
        raise RuntimeError(f"No GFS S3 keys found for {date_str} {cycle_str} in bucket {NOAA_GFS_S3_BUCKET}")

    logger.info("Found %d lead step(s) on AWS S3 for GFS run %s %sz", len(lead_keys), date_str, f"{chour:02d}")

    # Fetch lead steps in parallel
    lead_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_gfs_s3_single_lead, key, coords, s3): lead
            for lead, key in lead_keys.items()
        }
        for fut in as_completed(future_map):
            lead = future_map[fut]
            res = fut.result()
            if res:
                lead_data[lead] = res

    # Sort leads
    sorted_leads = sorted(lead_data.keys())
    if not sorted_leads:
        raise RuntimeError(f"Failed to fetch any lead steps from AWS S3 for GFS {date_str} {cycle_str}")

    # Meteorological de-accumulation for APCP:
    # GFS accumulation resets every 6 hours (0-1, 0-2..0-6, then 6-7, 6-8..6-12)
    # Step rain R_h = APCP_h - APCP_{h-1} if in same 6-hour bucket, else APCP_h
    deaccum_tp: Dict[int, List[float]] = {}
    for idx, lead in enumerate(sorted_leads):
        cur_acc = lead_data[lead]["apcp"]
        prev_lead = sorted_leads[idx - 1] if idx > 0 else None
        bucket_start = ((lead - 1) // 6) * 6

        if prev_lead is not None and prev_lead > bucket_start and prev_lead in lead_data:
            prev_acc = lead_data[prev_lead]["apcp"]
            deaccum_tp[lead] = [max(0.0, float(c - p)) for c, p in zip(cur_acc, prev_acc)]
        else:
            deaccum_tp[lead] = [max(0.0, float(c)) for c in cur_acc]

    records = []
    for lead in sorted_leads:
        valid_time = run_time + pd.Timedelta(hours=lead)
        d = lead_data[lead]
        rain_vals = deaccum_tp[lead]

        for loc_idx, (lat, lon) in enumerate(coords):
            gid = generate_grid_id(lat, lon)
            records.append({
                "origin": "gfs",
                "run_time": run_time,
                "valid_time": valid_time,
                "lead_time_hours": lead,
                "grid_id": gid,
                "lat": float(lat),
                "lon": float(lon),
                "tp": round(rain_vals[loc_idx], 2),
                "sp": round(float(d["sp"][loc_idx]), 2),
                "t2m": round(float(d["t2m"][loc_idx]), 2),
                "u10": round(float(d["u10"][loc_idx]), 3),
                "v10": round(float(d["v10"][loc_idx]), 3),
                "cape": round(float(d["cape"][loc_idx]), 1),
            })

    df = pd.DataFrame(records)
    logger.info("Successfully processed GFS from AWS S3 (%d records across %d leads)", len(df), len(sorted_leads))
    return df


# =============================================================================
# PART 2: OPEN-METEO ECMWF IFS (AND FALLBACK GFS) DOWNLOADER
# =============================================================================

def fetch_open_meteo_batch(
    locations: List[Tuple[float, float]],
    model_key: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Dict[str, Any]]:
    """Fetches historical forecast time series from Open-Meteo for a batch of locations."""
    model_cfg = SUPPORTED_HYBRID_MODELS[model_key]
    om_model = model_cfg["open_meteo_model"]

    lats_str = ",".join(f"{lat:.4f}" for lat, _ in locations)
    lons_str = ",".join(f"{lon:.4f}" for _, lon in locations)

    hourly_vars = [
        "precipitation",
        "rain",
        "temperature_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "cape",
    ]

    params = {
        "latitude": lats_str,
        "longitude": lons_str,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "models": om_model,
        "wind_speed_unit": "ms",
    }

    headers = {
        "User-Agent": "WeatherForecastEnhance/1.0 (HybridDownloader)",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else [data]
            elif resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("Open-Meteo HTTP %d on attempt %d/%d. Waiting %.1fs...",
                               resp.status_code, attempt, max_retries, retry_delay)
                time.sleep(retry_delay * attempt)
            else:
                resp.raise_for_status()
        except Exception as exc:
            last_error = exc
            logger.warning("Request error on attempt %d/%d: %s. Retrying in %.1fs...",
                           attempt, max_retries, exc, retry_delay)
            time.sleep(retry_delay * attempt)

    raise RuntimeError(f"Failed to fetch data from Open-Meteo after {max_retries} attempts: {last_error}")


def map_rolling_window_to_cycles(
    hourly_data: Dict[str, Any],
    lat: float,
    lon: float,
    origin: str,
    cycles: List[str] = DEFAULT_CYCLES,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
) -> pd.DataFrame:
    """Maps continuous hourly series from Open-Meteo into discrete NWP forecast cycles via Rolling-Window."""
    hourly_dict = hourly_data.get("hourly", {})
    if not hourly_dict or "time" not in hourly_dict:
        return pd.DataFrame()

    df_raw = pd.DataFrame(hourly_dict)
    if df_raw.empty or "time" not in df_raw.columns:
        return pd.DataFrame()

    df_raw["time"] = pd.to_datetime(df_raw["time"])

    precip_col = None
    for c in ["precipitation", "rain", f"precipitation_{origin}", f"rain_{origin}"]:
        if c in df_raw.columns and df_raw[c].notna().any():
            precip_col = c
            break

    temp_col = "temperature_2m" if "temperature_2m" in df_raw.columns else f"temperature_2m_{origin}"
    press_col = "surface_pressure" if "surface_pressure" in df_raw.columns else f"surface_pressure_{origin}"
    ws_col = "wind_speed_10m" if "wind_speed_10m" in df_raw.columns else f"wind_speed_10m_{origin}"
    wd_col = "wind_direction_10m" if "wind_direction_10m" in df_raw.columns else f"wind_direction_10m_{origin}"
    cape_col = "cape" if "cape" in df_raw.columns else f"cape_{origin}"

    df_raw = df_raw.set_index("time")
    min_time = df_raw.index.min()
    max_time = df_raw.index.max()

    unique_dates = pd.date_range(min_time.floor("D"), max_time.floor("D"), freq="D")
    records = []
    grid_id = generate_grid_id(lat, lon)

    for dt in unique_dates:
        for cycle_str in cycles:
            chour, cminute = [int(p) for p in cycle_str.split(":")]
            run_time = dt + pd.Timedelta(hours=chour, minutes=cminute)

            for lead in range(1, max_lead_hours + 1):
                valid_time = run_time + pd.Timedelta(hours=lead)
                if valid_time not in df_raw.index:
                    continue

                row = df_raw.loc[valid_time]
                tp_val = float(row[precip_col]) if precip_col and pd.notna(row[precip_col]) else 0.0
                t2m_val = float(row[temp_col]) if temp_col in row and pd.notna(row[temp_col]) else 25.0
                sp_val = float(row[press_col]) if press_col in row and pd.notna(row[press_col]) else 1010.0
                ws_val = float(row[ws_col]) if ws_col in row and pd.notna(row[ws_col]) else 0.0
                wd_val = float(row[wd_col]) if wd_col in row and pd.notna(row[wd_col]) else 0.0
                cape_val = float(row[cape_col]) if cape_col in row and pd.notna(row[cape_col]) else 0.0

                u10, v10 = compute_uv_components(ws_val, wd_val)

                records.append({
                    "origin": origin,
                    "run_time": run_time,
                    "valid_time": valid_time,
                    "lead_time_hours": lead,
                    "grid_id": grid_id,
                    "lat": float(lat),
                    "lon": float(lon),
                    "tp": max(0.0, tp_val),
                    "sp": max(800.0, min(1050.0, sp_val)),
                    "t2m": max(-10.0, min(50.0, t2m_val)),
                    "u10": round(float(u10), 3),
                    "v10": round(float(v10), 3),
                    "cape": max(0.0, cape_val),
                })

    return pd.DataFrame(records)


def download_open_meteo_cycle(
    model_key: str,
    date_str: str,
    cycle_str: str,
    coords: List[Tuple[float, float]],
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Downloads and maps a single forecast cycle from Open-Meteo."""
    start_dt = pd.to_datetime(date_str)
    end_dt = start_dt + pd.Timedelta(days=math.ceil((max_lead_hours + 12) / 24.0))

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    target_run_time = pd.Timestamp(f"{date_str} {cycle_str}")
    all_dfs = []

    for i in range(0, len(coords), batch_size):
        chunk = coords[i: i + batch_size]
        responses = fetch_open_meteo_batch(
            locations=chunk,
            model_key=model_key,
            start_date=start_str,
            end_date=end_str,
        )
        for loc_idx, loc_resp in enumerate(responses):
            lat, lon = chunk[loc_idx] if loc_idx < len(chunk) else (loc_resp.get("latitude", 0.0), loc_resp.get("longitude", 0.0))
            cycle_df = map_rolling_window_to_cycles(
                hourly_data=loc_resp,
                lat=lat,
                lon=lon,
                origin=model_key,
                cycles=[cycle_str],
                max_lead_hours=max_lead_hours,
            )
            if not cycle_df.empty:
                cycle_filtered = cycle_df[cycle_df["run_time"] == target_run_time]
                if not cycle_filtered.empty:
                    all_dfs.append(cycle_filtered)
        time.sleep(0.1)

    if not all_dfs:
        raise RuntimeError(f"No records retrieved from Open-Meteo for {model_key} on {date_str} {cycle_str}")

    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# PART 3: UNIFIED HYBRID INTERFACE
# =============================================================================

def download_hybrid_cycle(
    model: str,
    date_str: str,
    cycle_str: str = "00:00",
    locations: Optional[List[Tuple[float, float]]] = None,
    output_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    gfs_source: str = "aws",
) -> Path:
    """
    Downloads and maps a single forecast cycle for the specified model:
    - GFS: Pulls from AWS S3 by default (or falls back to Open-Meteo)
    - ECMWF IFS: Pulls from Open-Meteo + Rolling Window lead time
    """
    model_key = resolve_model_key(model)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT_NWP_DIR
    model_out_dir = out_dir / model_key
    model_out_dir.mkdir(parents=True, exist_ok=True)

    cycle_clean = cycle_str.replace(":", "")[:2]
    out_file = model_out_dir / f"run_{date_str.replace('-', '')}_{cycle_clean}z.parquet"
    if out_file.exists():
        logger.info("[Cached] Hybrid cycle already exists: %s", out_file)
        return out_file

    coords = load_locations(meta_dir=meta_dir, custom_coords=locations)
    df_result = None

    if model_key == "gfs" and gfs_source.lower() == "aws":
        try:
            df_result = download_gfs_cycle_from_aws(
                date_str=date_str,
                cycle_str=cycle_str,
                coords=coords,
                max_lead_hours=max_lead_hours,
            )
        except Exception as e:
            logger.warning("AWS S3 GFS retrieval encountered an issue: %s. Falling back to Open-Meteo...", e)

    if df_result is None or df_result.empty:
        # ECMWF IFS or GFS fallback
        logger.info("Fetching [%s] via Open-Meteo Historical Forecast API...", model_key.upper())
        df_result = download_open_meteo_cycle(
            model_key=model_key,
            date_str=date_str,
            cycle_str=cycle_str,
            coords=coords,
            max_lead_hours=max_lead_hours,
        )

    df_result.to_parquet(out_file, index=False, compression="snappy")
    logger.info("Successfully created Hybrid NWP cycle artifact: %s (%d rows)", out_file, len(df_result))
    return out_file


def run_hybrid_batch(
    models: List[str],
    start_date: str,
    end_date: str,
    locations: Optional[List[Tuple[float, float]]] = None,
    cycles: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    gfs_source: str = "aws",
) -> List[Path]:
    """
    Executes batch download across a date range:
    - GFS: directly from AWS S3 (noaa-gfs-bdp-pds)
    - ECMWF IFS: from Open-Meteo Historical Forecast API
    """
    cycles = cycles or DEFAULT_CYCLES
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT_NWP_DIR
    coords = load_locations(meta_dir=meta_dir, custom_coords=locations)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_list = pd.date_range(start_dt, end_dt, freq="D").strftime("%Y-%m-%d").tolist()

    saved_files = []

    for raw_m in models:
        model_key = resolve_model_key(raw_m)
        logger.info("=== Running Hybrid Batch for [%s] (%s to %s, %d locations) ===",
                    model_key.upper(), start_date, end_date, len(coords))

        for date_s in date_list:
            for cycle_s in cycles:
                try:
                    f = download_hybrid_cycle(
                        model=model_key,
                        date_str=date_s,
                        cycle_str=cycle_s,
                        locations=coords,
                        output_dir=out_dir,
                        meta_dir=meta_dir,
                        max_lead_hours=max_lead_hours,
                        gfs_source=gfs_source,
                    )
                    if f and f.exists() and f not in saved_files:
                        saved_files.append(f)
                except Exception as exc:
                    logger.error("Failed cycle %s %s for %s: %s", date_s, cycle_s, model_key, exc)

    logger.info("Hybrid acquisition complete. Saved %d cycle file(s).", len(saved_files))
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid NWP Downloader (GFS from AWS S3 & ECMWF IFS from Open-Meteo)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ecmwf_ifs", "gfs"],
        help="NWP Models (Strictly supports only: 'ecmwf_ifs' and 'gfs')",
    )
    parser.add_argument("--start-date", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2021-01-02", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cycles", nargs="+", default=DEFAULT_CYCLES, help="Forecast cycles (e.g. 00:00 12:00)")
    parser.add_argument("--max-lead", type=int, default=DEFAULT_MAX_LEAD, help="Max lead time hours (default: 24)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_NWP_DIR), help="Output directory for processed parquet cycles")
    parser.add_argument("--meta-dir", type=str, default=str(DEFAULT_HII_META_DIR), help="HII metadata directory")
    parser.add_argument("--gfs-source", choices=["aws", "openmeteo"], default="aws", help="Data source for GFS (default: aws)")
    parser.add_argument("--lat", type=float, default=None, help="Single target latitude")
    parser.add_argument("--lon", type=float, default=None, help="Single target longitude")

    args = parser.parse_args()

    validated_models = [resolve_model_key(m) for m in args.models]
    custom_coords = [(args.lat, args.lon)] if args.lat is not None and args.lon is not None else None

    run_hybrid_batch(
        models=validated_models,
        start_date=args.start_date,
        end_date=args.end_date,
        locations=custom_coords,
        cycles=args.cycles,
        output_dir=Path(args.out_dir),
        meta_dir=Path(args.meta_dir),
        max_lead_hours=args.max_lead,
        gfs_source=args.gfs_source,
    )


if __name__ == "__main__":
    main()
