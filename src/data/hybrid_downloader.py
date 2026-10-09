"""
Hybrid NWP Downloader & Transformer:
Combines real operational archives from:
  1. NOAA GFS: Directly from AWS S3 ('s3://noaa-gfs-bdp-pds/') via unsigned byte-range GRIB2 queries
     - Real model run cycles (00z, 06z, 12z, 18z) and genuine lead times (f001 to f024)
     - Covers 2021-2025+ complete archive
  2. ECMWF IFS: From Open-Meteo Historical Forecast API
     - Seamless hourly series transformed with a Rolling-Window lead time mapping (1 to 24h)
  3. JMA GSM: From Open-Meteo Historical Forecast API
     - Japan Meteorological Agency Global Spectral Model transformed with Rolling-Window lead time mapping (1 to 24h)

Produces identical Analysis-Ready Parquet files matching the system schema:
  ['origin', 'run_time', 'valid_time', 'lead_time_hours', 'grid_id', 'lat', 'lon',
   'tp', 'sp', 't2m', 'u10', 'v10', 'cape']
"""

import os
import sys
import time
import json
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

# Rate Limiting & Pacing Configuration for Open-Meteo
# Open-Meteo free tier imposes a 600 calls/min (10 calls/s) limit and 5000 calls/hr.
# Default delay of 0.25s (~4 calls/s = 240 calls/min) stays well within limits.
DEFAULT_DELAY_BETWEEN_CALLS = float(os.getenv("OPEN_METEO_DELAY", "0.25"))

_GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL = 0.0
_CURRENT_ADAPTIVE_DELAY = DEFAULT_DELAY_BETWEEN_CALLS

# Supported models
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
    "jma_gsm": {
        "primary_source": "openmeteo",
        "open_meteo_model": "jma_gsm",
        "origin_name": "jma_gsm",
        "description": "JMA GSM (Japan Meteorological Agency Global Spectral Model via Open-Meteo)",
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
    "jma_gsm": "jma_gsm",
    "jma": "jma_gsm",
    "rjtd": "jma_gsm",
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
    """Validates and resolves requested model name to canonical key ('ecmwf_ifs', 'gfs', or 'jma_gsm')."""
    m_clean = model.strip().lower()
    if m_clean in MODEL_ALIASES:
        return MODEL_ALIASES[m_clean]
    raise ValueError(
        f"Model '{model}' is not supported by hybrid downloader. "
        f"Hybrid downloader supports: 'ecmwf_ifs', 'gfs', and 'jma_gsm' "
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
    raw_dir: Optional[Path] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Downloads full forecast horizon of a GFS cycle from NOAA AWS S3
    and formats into analysis-ready records with genuine lead_time_hours.
    Persists raw per-lead checkpoints as JSON to allow resuming if stopped midway.
    """
    s3 = get_s3_client()
    chour = int(cycle_str.split(":")[0])
    run_time = pd.Timestamp(f"{date_str} {cycle_str}")
    date_clean = date_str.replace("-", "")
    cycle_clean = f"{chour:02d}z"

    raw_base = Path(raw_dir) if raw_dir else DEFAULT_RAW_NWP_DIR
    raw_gfs_dir = raw_base / "gfs" / f"run_{date_clean}_{cycle_clean}"
    raw_gfs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Ingesting GFS directly from AWS S3 (noaa-gfs-bdp-pds) for %s %s across %d location(s)...",
                date_str, cycle_clean, len(coords))

    # Resolve S3 keys for available lead steps
    lead_keys = {}
    for lead in range(1, max_lead_hours + 1):
        k = resolve_gfs_s3_key(date_str, chour, lead, s3_client=s3)
        if k:
            lead_keys[lead] = k

    if not lead_keys:
        raise RuntimeError(f"No GFS S3 keys found for {date_str} {cycle_str} in bucket {NOAA_GFS_S3_BUCKET}")

    logger.info("Found %d lead step(s) on AWS S3 for GFS run %s %s", len(lead_keys), date_str, cycle_clean)

    # Check for existing raw checkpoints on disk
    lead_data: Dict[int, Dict[str, List[float]]] = {}
    missing_leads: Dict[int, str] = {}

    for lead, key in lead_keys.items():
        lead_cache_file = raw_gfs_dir / f"lead_f{lead:03d}.json"
        if lead_cache_file.exists() and lead_cache_file.stat().st_size > 0:
            try:
                with open(lead_cache_file, "r", encoding="utf-8") as f:
                    cached_lead = json.load(f)
                cached_vars = cached_lead.get("variables", {})
                if all(v in cached_vars and len(cached_vars[v]) == len(coords)
                       for v in ["t2m", "sp", "u10", "v10", "apcp", "cape"]):
                    lead_data[lead] = cached_vars
                    continue
            except Exception as e:
                logger.warning("Corrupted GFS raw lead checkpoint %s (%s). Re-fetching...", lead_cache_file.name, e)

        missing_leads[lead] = key

    if lead_data:
        logger.info("[Cached Raw] Loaded %d/%d GFS lead step(s) from %s", len(lead_data), len(lead_keys), raw_gfs_dir)

    if missing_leads:
        logger.info("Fetching remaining %d/%d GFS lead step(s) from AWS S3...", len(missing_leads), len(lead_keys))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(fetch_gfs_s3_single_lead, key, coords, s3): (lead, key)
                for lead, key in missing_leads.items()
            }
            for fut in as_completed(future_map):
                lead, key = future_map[fut]
                res = fut.result()
                if res:
                    lead_data[lead] = res
                    # Save raw lead checkpoint atomically
                    lead_cache_file = raw_gfs_dir / f"lead_f{lead:03d}.json"
                    tmp_file = lead_cache_file.with_suffix(".tmp")
                    lead_payload = {
                        "date": date_str,
                        "cycle": cycle_str,
                        "lead": lead,
                        "s3_key": key,
                        "coords_count": len(coords),
                        "variables": res,
                    }
                    try:
                        with open(tmp_file, "w", encoding="utf-8") as f:
                            json.dump(lead_payload, f)
                        tmp_file.replace(lead_cache_file)
                    except Exception as e:
                        logger.warning("Could not write GFS lead checkpoint %s: %s", lead_cache_file.name, e)

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

def fetch_open_meteo_single(
    lat: float,
    lon: float,
    model_key: str,
    start_date: str,
    end_date: str,
    raw_dir: Optional[Path] = None,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """
    Fetches historical forecast time series from Open-Meteo for a single location (1 API call = 1 coordinate).
    Persists raw responses as JSON files to allow resuming if stopped midway.
    Handles HTTP 429 (Rate Limits) with Retry-After inspection, global cooldown, and exponential backoff.
    """
    global _GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL, _CURRENT_ADAPTIVE_DELAY

    raw_base = Path(raw_dir) if raw_dir else DEFAULT_RAW_NWP_DIR
    raw_om_dir = raw_base / "openmeteo" / model_key
    raw_om_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_om_dir / f"{model_key}_lat{lat:+.4f}_lon{lon:+.4f}_{start_date}_{end_date}.json"

    # Checkpoint check: if raw JSON exists and is non-empty, load from disk
    if raw_file.exists() and raw_file.stat().st_size > 0:
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if isinstance(cached_data, dict) and "hourly" in cached_data and cached_data["hourly"].get("time"):
                logger.info("[Cached Raw] Loaded Open-Meteo JSON for (%.4f, %.4f) from %s", lat, lon, raw_file.name)
                return cached_data
        except Exception as e:
            logger.warning("Corrupted raw Open-Meteo cache %s (%s). Re-fetching...", raw_file.name, e)

    model_cfg = SUPPORTED_HYBRID_MODELS[model_key]
    om_model = model_cfg["open_meteo_model"]

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
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
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
        # Enforce global cooldown if an earlier request hit a rate limit
        now = time.time()
        if _GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL > now:
            pause_time = _GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL - now
            logger.info(
                "Global rate-limit cooldown active. Pausing for %.1fs before querying (%.4f, %.4f)...",
                pause_time, lat, lon
            )
            time.sleep(pause_time)

        try:
            resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    data = data[0] if data else {}

                # Save raw JSON checkpoint atomically
                tmp_file = raw_file.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                tmp_file.replace(raw_file)
                logger.debug("Saved Open-Meteo raw JSON checkpoint: %s", raw_file.name)
                return data

            elif resp.status_code == 429:
                # Parse rate limit reason and retry duration
                err_text = resp.text.strip()
                err_reason = ""
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict):
                        err_reason = err_json.get("reason", "")
                except Exception:
                    pass
                reason_str = err_reason or err_text[:200]
                last_error = f"HTTP 429: {reason_str}"

                # Check Retry-After header
                wait_sec = None
                retry_after_hdr = resp.headers.get("Retry-After")
                if retry_after_hdr:
                    try:
                        wait_sec = float(retry_after_hdr) + 1.0
                    except (ValueError, TypeError):
                        pass

                if wait_sec is None:
                    lower_msg = reason_str.lower()
                    if "minute" in lower_msg:
                        # Minutely quota reset: wait full 60s
                        wait_sec = max(60.0, retry_delay * (2 ** (attempt - 1)))
                    elif "hour" in lower_msg:
                        logger.error(
                            "Open-Meteo Hourly Limit Exceeded (5,000 calls): %s. "
                            "Please wait for the next hour window.",
                            reason_str
                        )
                        wait_sec = max(60.0, retry_delay * (2 ** (attempt - 1)))
                    elif "daily" in lower_msg or "day" in lower_msg:
                        logger.error(
                            "Open-Meteo Daily Limit Exceeded (10,000 calls): %s. "
                            "Please wait for daily quota to reset.",
                            reason_str
                        )
                        wait_sec = max(60.0, retry_delay * (2 ** (attempt - 1)))
                    else:
                        # Exponential backoff for 429: 15s, 30s, 60s, 90s, 120s
                        wait_sec = max(15.0, retry_delay * (2 ** (attempt - 1)))

                # Enforce global cooldown so subsequent requests don't collide
                _GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL = max(_GLOBAL_RATE_LIMIT_COOLDOWN_UNTIL, time.time() + wait_sec)
                # Dynamically increase baseline session delay to prevent immediate rate limit re-triggers
                _CURRENT_ADAPTIVE_DELAY = max(_CURRENT_ADAPTIVE_DELAY, 0.5)

                logger.warning(
                    "Open-Meteo HTTP 429 (Rate Limited) for (%.4f, %.4f) on attempt %d/%d. "
                    "Reason: '%s'. Backoff waiting %.1fs (session delay increased to %.2fs)...",
                    lat, lon, attempt, max_retries, reason_str, wait_sec, _CURRENT_ADAPTIVE_DELAY
                )
                time.sleep(wait_sec)

            elif resp.status_code in (500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                wait_sec = retry_delay * attempt
                logger.warning(
                    "Open-Meteo HTTP %d for (%.4f, %.4f) on attempt %d/%d. Retrying in %.1fs...",
                    resp.status_code, lat, lon, attempt, max_retries, wait_sec
                )
                time.sleep(wait_sec)

            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                resp.raise_for_status()

        except Exception as exc:
            if not last_error or not str(last_error).startswith("HTTP"):
                last_error = exc
            wait_sec = retry_delay * attempt
            logger.warning(
                "Open-Meteo request error for (%.4f, %.4f) on attempt %d/%d: %s. Retrying in %.1fs...",
                lat, lon, attempt, max_retries, exc, wait_sec
            )
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed to fetch data from Open-Meteo for ({lat:.4f}, {lon:.4f}) after {max_retries} attempts: {last_error}")


def map_rolling_window_to_cycles(
    hourly_data: Dict[str, Any],
    lat: float,
    lon: float,
    origin: str,
    cycles: List[str] = DEFAULT_CYCLES,
    target_dates: Optional[List[str]] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
) -> pd.DataFrame:
    """
    Maps continuous hourly series from Open-Meteo into discrete NWP forecast cycles via Rolling-Window.
    High-performance vectorized implementation (sub-50ms per multi-year series).
    """
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

    if target_dates is not None and len(target_dates) > 0:
        unique_dates = pd.to_datetime(target_dates)
    else:
        unique_dates = pd.date_range(min_time.floor("D"), max_time.floor("D"), freq="D")

    run_times_list = []
    for dt in unique_dates:
        for c in cycles:
            chour, cminute = [int(p) for p in c.split(":")]
            run_times_list.append(dt + pd.Timedelta(hours=chour, minutes=cminute))

    if not run_times_list:
        return pd.DataFrame()

    run_times = pd.DatetimeIndex(run_times_list)
    leads = np.arange(1, max_lead_hours + 1)
    n_runs = len(run_times)

    run_time_arr = np.repeat(run_times.values, max_lead_hours)
    lead_arr = np.tile(leads, n_runs)
    valid_time_arr = run_time_arr + pd.to_timedelta(lead_arr, unit="h").values

    valid_dt = pd.DatetimeIndex(valid_time_arr)
    in_index_mask = valid_dt.isin(df_raw.index)

    if not np.all(in_index_mask):
        run_time_arr = run_time_arr[in_index_mask]
        lead_arr = lead_arr[in_index_mask]
        valid_time_arr = valid_time_arr[in_index_mask]
        valid_dt = valid_dt[in_index_mask]

    if len(valid_dt) == 0:
        return pd.DataFrame()

    aligned_df = df_raw.reindex(valid_dt)

    tp_vals = np.maximum(0.0, aligned_df[precip_col].fillna(0.0).values) if precip_col and precip_col in aligned_df else np.zeros(len(valid_dt))
    sp_vals = np.clip(aligned_df[press_col].fillna(1010.0).values, 800.0, 1050.0) if press_col in aligned_df else np.full(len(valid_dt), 1010.0)
    t2m_vals = np.clip(aligned_df[temp_col].fillna(25.0).values, -10.0, 50.0) if temp_col in aligned_df else np.full(len(valid_dt), 25.0)
    ws_vals = aligned_df[ws_col].fillna(0.0).values if ws_col in aligned_df else np.zeros(len(valid_dt))
    wd_vals = aligned_df[wd_col].fillna(0.0).values if wd_col in aligned_df else np.zeros(len(valid_dt))
    cape_vals = np.maximum(0.0, aligned_df[cape_col].fillna(0.0).values) if cape_col in aligned_df else np.zeros(len(valid_dt))

    u10_vals, v10_vals = compute_uv_components(ws_vals, wd_vals)

    grid_id = generate_grid_id(lat, lon)
    return pd.DataFrame({
        "origin": origin,
        "run_time": run_time_arr,
        "valid_time": valid_time_arr,
        "lead_time_hours": lead_arr,
        "grid_id": grid_id,
        "lat": float(round(lat, 4)),
        "lon": float(round(lon, 4)),
        "tp": np.round(tp_vals, 3),
        "sp": np.round(sp_vals, 1),
        "t2m": np.round(t2m_vals, 2),
        "u10": np.round(u10_vals, 3),
        "v10": np.round(v10_vals, 3),
        "cape": np.round(cape_vals, 1),
    })


def download_open_meteo_cycle(
    model_key: str,
    date_str: str,
    cycle_str: str,
    coords: List[Tuple[float, float]],
    raw_dir: Optional[Path] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    delay_between_calls: Optional[float] = None,
) -> pd.DataFrame:
    """
    Downloads and maps a single forecast cycle from Open-Meteo.
    Strictly performs 1 API call per single coordinate pair (1 call 1 พิกัด).
    Persists raw responses as JSON to raw_dir to allow resuming if stopped midway.
    """
    start_dt = pd.to_datetime(date_str)
    end_dt = start_dt + pd.Timedelta(days=math.ceil((max_lead_hours + 12) / 24.0))

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    target_run_time = pd.Timestamp(f"{date_str} {cycle_str}")
    all_dfs = []
    actual_delay = delay_between_calls if delay_between_calls is not None else DEFAULT_DELAY_BETWEEN_CALLS

    logger.info("Fetching [%s] via Open-Meteo for cycle %s %s (1 call per location, total %d locations, pacing %.2fs)...",
                model_key.upper(), date_str, cycle_str, len(coords), actual_delay)

    for idx, (lat, lon) in enumerate(coords, start=1):
        try:
            loc_resp = fetch_open_meteo_single(
                lat=lat,
                lon=lon,
                model_key=model_key,
                start_date=start_str,
                end_date=end_str,
                raw_dir=raw_dir,
            )
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
        except Exception as exc:
            logger.warning("Failed Open-Meteo fetch for location (%.4f, %.4f) on %s %s: %s",
                           lat, lon, date_str, cycle_str, exc)

        current_delay = max(actual_delay, _CURRENT_ADAPTIVE_DELAY)
        if current_delay > 0 and idx < len(coords):
            time.sleep(current_delay)

    if not all_dfs:
        raise RuntimeError(f"No records retrieved from Open-Meteo for {model_key} on {date_str} {cycle_str}")

    return pd.concat(all_dfs, ignore_index=True)


def download_open_meteo_date_range(
    model_key: str,
    start_date: str,
    end_date: str,
    cycles: List[str],
    coords: List[Tuple[float, float]],
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    delay_between_calls: Optional[float] = None,
    chunk_days: int = 30,
) -> List[Path]:
    """
    Downloads and maps Open-Meteo forecasts across a date range.
    Queries Open-Meteo strictly 1 API call per single coordinate pair (1 call 1 พิกัด)
    for the requested date window with raw JSON caching to raw_dir.
    Processes and writes Parquet cycle files incrementally in batches of chunk_days (default 30 days)
    to provide real-time saving to disk/Google Drive and prevent memory exhaustion (OOM).
    """
    model_out_dir = output_dir / model_key
    model_out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    api_end_dt = end_dt + pd.Timedelta(days=math.ceil((max_lead_hours + 12) / 24.0))

    start_str = start_dt.strftime("%Y-%m-%d")
    api_end_str = api_end_dt.strftime("%Y-%m-%d")

    date_list = pd.date_range(start_dt, end_dt, freq="D").strftime("%Y-%m-%d").tolist()

    needed_cycles = []
    for d_str in date_list:
        for c_str in cycles:
            cycle_clean = c_str.replace(":", "")[:2]
            out_file = model_out_dir / f"run_{d_str.replace('-', '')}_{cycle_clean}z.parquet"
            if not out_file.exists():
                needed_cycles.append((d_str, c_str, out_file))

    if not needed_cycles:
        logger.info("[Cached] All %d Open-Meteo cycle file(s) for [%s] already exist in %s",
                    len(date_list) * len(cycles), model_key.upper(), model_out_dir)
        return [
            model_out_dir / f"run_{d_str.replace('-', '')}_{c_str.replace(':', '')[:2]}z.parquet"
            for d_str in date_list for c_str in cycles
        ]

    actual_delay = delay_between_calls if delay_between_calls is not None else DEFAULT_DELAY_BETWEEN_CALLS
    total_locs = len(coords)
    logger.info("Ingesting [%s] via Open-Meteo (1 call per location, %d location(s), %s to %s, pacing %.2fs)...",
                model_key.upper(), total_locs, start_str, end_date, actual_delay)

    # Phase 1: Ensure all locations have raw JSON downloaded and checkpointed to raw_dir
    for idx, (lat, lon) in enumerate(coords, start=1):
        try:
            fetch_open_meteo_single(
                lat=lat,
                lon=lon,
                model_key=model_key,
                start_date=start_str,
                end_date=api_end_str,
                raw_dir=raw_dir,
            )
            if idx % 50 == 1 or idx == total_locs or total_locs <= 10:
                pct = (idx / total_locs) * 100
                logger.info("[%s Raw Checkpoint: %d/%d (%.1f%%)] Loaded/Fetched (%.4f, %.4f)",
                            model_key.upper(), idx, total_locs, pct, lat, lon)
        except Exception as exc:
            logger.warning("Failed Open-Meteo fetch for location (%.4f, %.4f): %s", lat, lon, exc)

        current_delay = max(actual_delay, _CURRENT_ADAPTIVE_DELAY)
        if current_delay > 0 and idx < total_locs:
            time.sleep(current_delay)

    # Phase 2: Incrementally generate and write Parquet cycles in chunks of chunk_days
    # This guarantees immediate file creation on Google Drive / local disk and prevents RAM exhaustion
    chunks = [date_list[i:i + chunk_days] for i in range(0, len(date_list), chunk_days)]
    saved_files = []
    total_chunks = len(chunks)

    logger.info("Converting [%s] raw checkpoints to Analysis-Ready Parquets in %d date chunk(s)...",
                model_key.upper(), total_chunks)

    for chunk_idx, chunk_dates in enumerate(chunks, start=1):
        # Check if any cycles in this chunk are missing
        chunk_needed = []
        for d_str in chunk_dates:
            for c_str in cycles:
                cycle_clean = c_str.replace(":", "")[:2]
                out_file = model_out_dir / f"run_{d_str.replace('-', '')}_{cycle_clean}z.parquet"
                if not out_file.exists():
                    chunk_needed.append((d_str, c_str, out_file))
                elif out_file not in saved_files:
                    saved_files.append(out_file)

        if not chunk_needed:
            continue

        chunk_dfs = []
        for lat, lon in coords:
            try:
                loc_resp = fetch_open_meteo_single(
                    lat=lat,
                    lon=lon,
                    model_key=model_key,
                    start_date=start_str,
                    end_date=api_end_str,
                    raw_dir=raw_dir,
                )
                df_loc = map_rolling_window_to_cycles(
                    hourly_data=loc_resp,
                    lat=lat,
                    lon=lon,
                    origin=model_key,
                    cycles=cycles,
                    target_dates=chunk_dates,
                    max_lead_hours=max_lead_hours,
                )
                if not df_loc.empty:
                    chunk_dfs.append(df_loc)
            except Exception as exc:
                logger.warning("Could not map chunk data for (%.4f, %.4f): %s", lat, lon, exc)

        if not chunk_dfs:
            continue

        df_chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
        chunk_saved = 0

        for d_str, c_str, out_file in chunk_needed:
            target_run_time = pd.Timestamp(f"{d_str} {c_str}")
            cycle_data = df_chunk_combined[df_chunk_combined["run_time"] == target_run_time]
            if not cycle_data.empty:
                cycle_data.to_parquet(out_file, index=False, compression="snappy")
                if out_file not in saved_files:
                    saved_files.append(out_file)
                chunk_saved += 1
            elif out_file.exists() and out_file not in saved_files:
                saved_files.append(out_file)

        logger.info(
            "[%s Parquet Chunk %d/%d (%.1f%%)] Saved %d cycle files (%s to %s) in %s",
            model_key.upper(), chunk_idx, total_chunks, (chunk_idx / total_chunks) * 100,
            chunk_saved, chunk_dates[0], chunk_dates[-1], model_out_dir
        )
        del chunk_dfs, df_chunk_combined

    return saved_files


# =============================================================================
# PART 3: UNIFIED HYBRID INTERFACE
# =============================================================================

def download_hybrid_cycle(
    model: str,
    date_str: str,
    cycle_str: str = "00:00",
    locations: Optional[List[Tuple[float, float]]] = None,
    output_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    stations_limit: Optional[int] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    gfs_source: str = "aws",
    delay_between_calls: Optional[float] = None,
) -> Path:
    """
    Downloads and maps a single forecast cycle for the specified model:
    - GFS: Pulls from AWS S3 by default (or falls back to Open-Meteo) with raw checkpointing
    - ECMWF IFS / JMA GSM: Pulls from Open-Meteo (1 call per location, raw JSON cached) + Rolling Window lead time
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

    coords = load_locations(meta_dir=meta_dir, limit=stations_limit, custom_coords=locations)
    df_result = None

    if model_key == "gfs" and gfs_source.lower() == "aws":
        try:
            df_result = download_gfs_cycle_from_aws(
                date_str=date_str,
                cycle_str=cycle_str,
                coords=coords,
                raw_dir=raw_dir,
                max_lead_hours=max_lead_hours,
            )
        except Exception as e:
            logger.warning("AWS S3 GFS retrieval encountered an issue: %s. Falling back to Open-Meteo...", e)

    if df_result is None or df_result.empty:
        # ECMWF IFS, JMA GSM, or GFS fallback
        logger.info("Fetching [%s] via Open-Meteo Historical Forecast API (1 call per location)...", model_key.upper())
        df_result = download_open_meteo_cycle(
            model_key=model_key,
            date_str=date_str,
            cycle_str=cycle_str,
            coords=coords,
            raw_dir=raw_dir,
            max_lead_hours=max_lead_hours,
            delay_between_calls=delay_between_calls,
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
    raw_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    stations_limit: Optional[int] = None,
    max_lead_hours: int = DEFAULT_MAX_LEAD,
    gfs_source: str = "aws",
    delay_between_calls: Optional[float] = None,
) -> List[Path]:
    """
    Executes batch download across a date range:
    - GFS: directly from AWS S3 (noaa-gfs-bdp-pds) with raw lead checkpoints to allow resuming
    - ECMWF IFS / JMA GSM: strictly 1 call per single location from Open-Meteo with raw JSON checkpointing
    Tracks overall progress percentage, per-NWP-model breakdown, and records skipped cached files.
    """
    cycles = cycles or DEFAULT_CYCLES
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT_NWP_DIR
    coords = load_locations(meta_dir=meta_dir, limit=stations_limit, custom_coords=locations)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_list = pd.date_range(start_dt, end_dt, freq="D").strftime("%Y-%m-%d").tolist()

    validated_models = [resolve_model_key(m) for m in models]
    cycles_per_model = len(date_list) * len(cycles)
    total_cycles_all_models = len(validated_models) * cycles_per_model

    overall_done = 0
    overall_downloaded = 0
    overall_skipped = 0

    model_totals = {m: cycles_per_model for m in validated_models}
    model_done = {m: 0 for m in validated_models}
    model_downloaded = {m: 0 for m in validated_models}
    model_skipped = {m: 0 for m in validated_models}

    skipped_files: List[Dict[str, Any]] = []
    downloaded_files: List[Dict[str, Any]] = []
    saved_files: List[Path] = []

    logger.info("Initiating Hybrid NWP Acquisition: %d total cycle(s) across %d model(s) (%s to %s, %d locations)",
                total_cycles_all_models, len(validated_models), start_date, end_date, len(coords))
    for m in validated_models:
        logger.info("  * NWP Model Target: [%s] -> %d cycle(s)", m.upper(), cycles_per_model)

    for model_key in validated_models:
        model_out_dir = out_dir / model_key
        model_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== Processing Hybrid Batch for [%s] (%s to %s) ===",
                    model_key.upper(), start_date, end_date)

        if model_key == "gfs" and gfs_source.lower() == "aws":
            for date_s in date_list:
                for cycle_s in cycles:
                    cycle_clean = cycle_s.replace(":", "")[:2]
                    target_file = model_out_dir / f"run_{date_s.replace('-', '')}_{cycle_clean}z.parquet"

                    if target_file.exists():
                        overall_done += 1
                        overall_skipped += 1
                        model_done[model_key] += 1
                        model_skipped[model_key] += 1
                        skipped_files.append({
                            "path": target_file,
                            "name": target_file.name,
                            "model": model_key,
                            "date": date_s,
                            "cycle": cycle_s,
                        })
                        if target_file not in saved_files:
                            saved_files.append(target_file)

                        overall_pct = (overall_done / total_cycles_all_models) * 100
                        m_pct = (model_done[model_key] / model_totals[model_key]) * 100
                        logger.info(
                            "[Progress: %d/%d (%.1f%%) | %s: %d/%d (%.1f%%)] [SKIP CACHED] Skipped existing cycle: %s",
                            overall_done, total_cycles_all_models, overall_pct,
                            model_key.upper(), model_done[model_key], model_totals[model_key], m_pct,
                            target_file.name
                        )
                        continue

                    # Download single GFS cycle
                    overall_pct_start = ((overall_done + 1) / total_cycles_all_models) * 100
                    m_pct_start = ((model_done[model_key] + 1) / model_totals[model_key]) * 100
                    logger.info(
                        "[Progress: %d/%d (%.1f%%) | %s: %d/%d (%.1f%%)] [DOWNLOADING] Fetching GFS cycle %s %s...",
                        overall_done + 1, total_cycles_all_models, overall_pct_start,
                        model_key.upper(), model_done[model_key] + 1, model_totals[model_key], m_pct_start,
                        date_s, cycle_s
                    )
                    try:
                        f = download_hybrid_cycle(
                            model=model_key,
                            date_str=date_s,
                            cycle_str=cycle_s,
                            locations=coords,
                            output_dir=out_dir,
                            raw_dir=raw_dir,
                            meta_dir=meta_dir,
                            stations_limit=stations_limit,
                            max_lead_hours=max_lead_hours,
                            gfs_source=gfs_source,
                            delay_between_calls=delay_between_calls,
                        )
                        if f and f.exists():
                            overall_done += 1
                            overall_downloaded += 1
                            model_done[model_key] += 1
                            model_downloaded[model_key] += 1
                            downloaded_files.append({
                                "path": f,
                                "name": f.name,
                                "model": model_key,
                                "date": date_s,
                                "cycle": cycle_s,
                            })
                            if f not in saved_files:
                                saved_files.append(f)
                    except Exception as exc:
                        logger.error("Failed cycle %s %s for %s: %s", date_s, cycle_s, model_key, exc)
        else:
            # Open-Meteo models (ECMWF IFS, JMA GSM, or fallback GFS)
            # Pre-scan existing files to detect skipped vs needed
            needed_cycles = []
            for date_s in date_list:
                for cycle_s in cycles:
                    cycle_clean = cycle_s.replace(":", "")[:2]
                    target_file = model_out_dir / f"run_{date_s.replace('-', '')}_{cycle_clean}z.parquet"

                    if target_file.exists():
                        overall_done += 1
                        overall_skipped += 1
                        model_done[model_key] += 1
                        model_skipped[model_key] += 1
                        skipped_files.append({
                            "path": target_file,
                            "name": target_file.name,
                            "model": model_key,
                            "date": date_s,
                            "cycle": cycle_s,
                        })
                        if target_file not in saved_files:
                            saved_files.append(target_file)

                        overall_pct = (overall_done / total_cycles_all_models) * 100
                        m_pct = (model_done[model_key] / model_totals[model_key]) * 100
                        logger.info(
                            "[Progress: %d/%d (%.1f%%) | %s: %d/%d (%.1f%%)] [SKIP CACHED] Skipped existing cycle: %s",
                            overall_done, total_cycles_all_models, overall_pct,
                            model_key.upper(), model_done[model_key], model_totals[model_key], m_pct,
                            target_file.name
                        )
                    else:
                        needed_cycles.append((date_s, cycle_s, target_file))

            if needed_cycles:
                logger.info(
                    "[%s] Downloading %d missing cycle(s) via Open-Meteo across %d locations...",
                    model_key.upper(), len(needed_cycles), len(coords)
                )
                try:
                    om_files = download_open_meteo_date_range(
                        model_key=model_key,
                        start_date=start_date,
                        end_date=end_date,
                        cycles=cycles,
                        coords=coords,
                        output_dir=out_dir,
                        raw_dir=raw_dir,
                        max_lead_hours=max_lead_hours,
                        delay_between_calls=delay_between_calls,
                    )
                    for f in om_files:
                        if f and f.exists() and f not in saved_files:
                            saved_files.append(f)
                            overall_done += 1
                            overall_downloaded += 1
                            model_done[model_key] += 1
                            model_downloaded[model_key] += 1
                            downloaded_files.append({
                                "path": f,
                                "name": f.name,
                                "model": model_key,
                                "date": "",
                                "cycle": "",
                            })
                except Exception as exc:
                    logger.error("Failed Open-Meteo date range download for %s: %s", model_key, exc)

    # Detailed Acquisition Summary Report Banner
    overall_completion_pct = (overall_done / total_cycles_all_models * 100) if total_cycles_all_models else 0.0
    dl_pct = (overall_downloaded / total_cycles_all_models * 100) if total_cycles_all_models else 0.0
    sk_pct = (overall_skipped / total_cycles_all_models * 100) if total_cycles_all_models else 0.0

    logger.info("================================================================================")
    logger.info("                         HYBRID NWP ACQUISITION SUMMARY                         ")
    logger.info("================================================================================")
    logger.info("Total Forecast Cycles Requested : %d", total_cycles_all_models)
    logger.info("Overall Completion Rate         : %.1f%% (%d/%d cycles processed)", overall_completion_pct, overall_done, total_cycles_all_models)
    logger.info("  - Downloaded / Generated      : %d (%.1f%%)", overall_downloaded, dl_pct)
    logger.info("  - Skipped (Already Cached)    : %d (%.1f%%)", overall_skipped, sk_pct)
    logger.info("--------------------------------------------------------------------------------")
    logger.info("Breakdown by NWP Model:")
    for m, m_tot in model_totals.items():
        m_d = model_done[m]
        m_dl = model_downloaded[m]
        m_sk = model_skipped[m]
        m_pct = (m_d / m_tot * 100) if m_tot else 0.0
        logger.info(
            "  * [%s]: %d/%d (%.1f%%) | Downloaded: %d | Skipped: %d",
            m.upper(), m_d, m_tot, m_pct, m_dl, m_sk
        )
    logger.info("--------------------------------------------------------------------------------")
    if skipped_files:
        logger.info("Skipped Files List (%d files):", len(skipped_files))
        max_display = 25
        for item in skipped_files[:max_display]:
            cycle_info = f" | {item['date']} {item['cycle']}" if item.get('date') else ""
            logger.info("  [SKIPPED] %s (Model: %s%s)", item["name"], item["model"].upper(), cycle_info)
        if len(skipped_files) > max_display:
            logger.info("  ... and %d more cached files skipped (total skipped: %d)",
                        len(skipped_files) - max_display, len(skipped_files))
    else:
        logger.info("No files were skipped (all cycles were newly downloaded/generated).")
    logger.info("================================================================================")

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid NWP Downloader (GFS from AWS S3, ECMWF IFS & JMA GSM from Open-Meteo)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ecmwf_ifs", "gfs"],
        help="NWP Models (Supports: 'ecmwf_ifs', 'gfs', 'jma_gsm')",
    )
    parser.add_argument(
        "--start-date", "--start",
        dest="start_date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", "--end",
        dest="end_date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sample-days",
        type=int,
        default=2,
        help="Number of sample days for dev mode (default: 2)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full multi-year production download (2021-01-01 to 2025-12-31 across all stations)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Dev sample mode (limited stations and sample days)",
    )
    parser.add_argument(
        "--stations-limit",
        type=int,
        default=None,
        help="Limit number of stations to download (default: None for full mode)",
    )
    parser.add_argument("--cycles", nargs="+", default=DEFAULT_CYCLES, help="Forecast cycles (e.g. 00:00 12:00)")
    parser.add_argument("--max-lead", type=int, default=DEFAULT_MAX_LEAD, help="Max lead time hours (default: 24)")
    parser.add_argument(
        "--delay", "--delay-between-calls",
        dest="delay_between_calls",
        type=float,
        default=None,
        help="Delay in seconds between Open-Meteo API requests (default: 0.25s)",
    )
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_NWP_DIR), help="Output directory for processed parquet cycles")
    parser.add_argument("--raw-dir", type=str, default=None, help="Output directory for raw checkpoints (defaults to {out-dir}/raw if out-dir is customized, otherwise DEFAULT_RAW_NWP_DIR)")
    parser.add_argument("--meta-dir", type=str, default=str(DEFAULT_HII_META_DIR), help="HII metadata directory")
    parser.add_argument("--gfs-source", choices=["aws", "openmeteo"], default="aws", help="Data source for GFS (default: aws)")
    parser.add_argument("--lat", type=float, default=None, help="Single target latitude")
    parser.add_argument("--lon", type=float, default=None, help="Single target longitude")

    args = parser.parse_args()

    validated_models = [resolve_model_key(m) for m in args.models]
    custom_coords = [(args.lat, args.lon)] if args.lat is not None and args.lon is not None else None

    # Date resolution protocol matching tigge_downloader & pipeline:
    start_dt = pd.Timestamp(args.start_date)
    if args.end_date:
        end_dt = pd.Timestamp(args.end_date)
    elif args.full:
        end_dt = pd.Timestamp("2025-12-31")
    else:
        end_dt = start_dt + pd.Timedelta(days=args.sample_days - 1)

    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    # Stations limit protocol:
    if args.stations_limit is not None:
        stn_limit = args.stations_limit
    elif args.full:
        stn_limit = None  # Full production mode -> all stations
    elif custom_coords:
        stn_limit = None
    else:
        stn_limit = 5  # Dev mode default to avoid unintended heavy loads

    out_dir_path = Path(args.out_dir)
    if args.raw_dir:
        raw_dir_path = Path(args.raw_dir)
    elif str(args.out_dir) != str(DEFAULT_OUT_NWP_DIR):
        raw_dir_path = out_dir_path / "raw"
    else:
        raw_dir_path = DEFAULT_RAW_NWP_DIR

    logger.info("Hybrid execution mode: %s (Dates: %s to %s, Stations limit: %s, Raw dir: %s)",
                "FULL PRODUCTION" if args.full else "DEV SAMPLE",
                start_date_str, end_date_str, stn_limit if stn_limit else "ALL", raw_dir_path)

    run_hybrid_batch(
        models=validated_models,
        start_date=start_date_str,
        end_date=end_date_str,
        locations=custom_coords,
        cycles=args.cycles,
        output_dir=out_dir_path,
        raw_dir=raw_dir_path,
        meta_dir=Path(args.meta_dir),
        stations_limit=stn_limit,
        max_lead_hours=args.max_lead,
        gfs_source=args.gfs_source,
        delay_between_calls=args.delay_between_calls,
    )


if __name__ == "__main__":
    main()
