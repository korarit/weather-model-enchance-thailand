"""
Himawari Satellite AWS S3 Downloader & Thailand Grid Extractor
Fetches real Himawari-8 & Himawari-9 AHI Level 1b (HSD) observation data from NOAA Open Data on AWS S3.

Data Coverage:
- 2021-01-01 to 2022-12-13 04:50 UTC: Himawari-8 (s3://noaa-himawari8/)
- 2022-12-13 05:00 UTC onwards (2023, 2024, 2025): Himawari-9 (s3://noaa-himawari9/)

Optimization:
- Downloads only segments covering Thailand (Segments 3, 4, 5) rather than the entire Full Disk
- Cuts bandwidth from ~800 MB down to ~6-9 MB per band/snapshot
- Automatic calibration to Brightness Temperature (Kelvin) via Satpy
"""

import os
import sys
import bz2
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import datetime
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice.*")

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from satpy import Scene

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# NOAA S3 Buckets
BUCKET_H8 = "noaa-himawari8"
BUCKET_H9 = "noaa-himawari9"

# Satellite Operational Cutover timestamp (Dec 13, 2022 ~05:00 UTC)
HIMAWARI_CUTOVER_UTC = pd.Timestamp("2022-12-13 05:00:00", tz="UTC")

# Default Thailand Spatial Coordinates Grid
GRID_LATS = np.arange(5.5, 21.0, 0.5)
GRID_LONS = np.arange(97.0, 106.0, 0.5)

# Segments covering Thailand (Lat 5.5 - 21.0 N)
# Segment 3: 32N-21N (covers Northern boundary)
# Segment 4: 21N-10N (covers Upper, Central, East, NE)
# Segment 5: 10N-0N   (covers Southern Thailand down to 5.5N)
DEFAULT_SEGS = [3, 4, 5]

DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "cache" / "himawari_s3"


def get_himawari_bucket_and_sat(target_dt: pd.Timestamp) -> Tuple[str, str]:
    """
    Determines whether to use Himawari-8 or Himawari-9 based on timestamp.
    """
    ts = target_dt
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    if ts < HIMAWARI_CUTOVER_UTC:
        return BUCKET_H8, "H08"
    else:
        return BUCKET_H9, "H09"


def round_to_himawari_10m(dt: pd.Timestamp) -> pd.Timestamp:
    """Rounds timestamp to the nearest 10-minute observation cycle."""
    minute = dt.minute
    rem = minute % 10
    if rem < 5:
        rounded_m = minute - rem
    else:
        rounded_m = minute + (10 - rem)
    
    if rounded_m == 60:
        return dt.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    else:
        return dt.replace(minute=rounded_m, second=0, microsecond=0)


def build_s3_prefix_and_keys(
    target_dt: pd.Timestamp,
    band: str = "B13",
    segments: Optional[List[int]] = None
) -> Tuple[str, List[str]]:
    """
    Constructs the S3 bucket name and list of keys for the requested observation.
    band: 'B13' or 'B08' (or 'band13', 'band08')
    """
    band_code = "B13" if "13" in band else "B08"
    segments = segments or DEFAULT_SEGS
    dt = round_to_himawari_10m(target_dt)
    bucket, sat_id = get_himawari_bucket_and_sat(dt)

    date_str = dt.strftime("%Y/%m/%d")
    hhmm = dt.strftime("%H%M")
    ymd = dt.strftime("%Y%m%d")

    prefix = f"AHI-L1b-FLDK/{date_str}/{hhmm}/"
    keys = [
        f"{prefix}HS_{sat_id}_{ymd}_{hhmm}_{band_code}_FLDK_R20_S{seg:02d}10.DAT.bz2"
        for seg in segments
    ]
    return bucket, keys


def download_and_extract_hsd_segments(
    bucket: str,
    keys: List[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    s3_client=None
) -> List[Path]:
    """
    Downloads and decompresses .DAT.bz2 segments into local cache directory.
    Returns the paths to the decompressed .DAT files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if s3_client is None:
        s3_client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))

    extracted_dats: List[Path] = []
    for k in keys:
        fname = Path(k).name
        dat_fname = fname[:-4] if fname.endswith(".bz2") else fname
        dat_path = cache_dir / dat_fname
        bz2_path = cache_dir / fname

        if dat_path.exists() and dat_path.stat().st_size > 0:
            extracted_dats.append(dat_path)
            continue

        # Download .bz2
        try:
            logger.debug("Downloading s3://%s/%s -> %s", bucket, k, bz2_path.name)
            s3_client.download_file(bucket, k, str(bz2_path))
        except Exception as e:
            logger.warning("Could not download s3://%s/%s: %s", bucket, k, e)
            if bz2_path.exists():
                bz2_path.unlink()
            continue

        # Decompress
        try:
            with bz2.open(bz2_path, "rb") as fin, open(dat_path, "wb") as fout:
                fout.write(fin.read())
            bz2_path.unlink()  # remove compressed file to save space
            extracted_dats.append(dat_path)
        except Exception as e:
            logger.warning("Failed decompressing %s: %s", bz2_path.name, e)
            if bz2_path.exists():
                bz2_path.unlink()
            if dat_path.exists():
                dat_path.unlink()

    return extracted_dats


def extract_satellite_band_to_grid(
    dat_files: List[Path],
    target_dt: pd.Timestamp,
    band: str = "band13",
    grid_lats: np.ndarray = GRID_LATS,
    grid_lons: np.ndarray = GRID_LONS
) -> pd.DataFrame:
    """
    Uses Satpy to load the HSD segments, reprojects/crops over Thailand,
    and calculates statistical aggregates for each 0.5-degree grid point.
    """
    if not dat_files:
        raise ValueError(f"No DAT files provided for extraction at {target_dt}")

    dataset_name = "B13" if "13" in band else "B08"
    str_files = [str(p) for p in dat_files]

    scn = Scene(filenames=str_files, reader="ahi_hsd")
    scn.load([dataset_name])
    ds = scn[dataset_name]

    area = ds.attrs["area"]
    lons, lats = area.get_lonlats()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_vals = ds.values

    # Filter pixels covering Thailand region (with 0.5 deg margin)
    lat_min, lat_max = float(grid_lats.min() - 0.5), float(grid_lats.max() + 0.5)
    lon_min, lon_max = float(grid_lons.min() - 0.5), float(grid_lons.max() + 0.5)

    valid_mask = (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max) &
        (~np.isnan(raw_vals))
    )

    p_lats = lats[valid_mask]
    p_lons = lons[valid_mask]
    p_vals = raw_vals[valid_mask]

    if len(p_vals) == 0:
        raise ValueError(f"No valid satellite pixels found over Thailand region for {target_dt}")

    # Map each pixel to the nearest grid point
    lat_idx = np.clip(np.round((p_lats - grid_lats[0]) / (grid_lats[1] - grid_lats[0])).astype(int), 0, len(grid_lats) - 1)
    lon_idx = np.clip(np.round((p_lons - grid_lons[0]) / (grid_lons[1] - grid_lons[0])).astype(int), 0, len(grid_lons) - 1)

    grid_assigned_lats = grid_lats[lat_idx]
    grid_assigned_lons = grid_lons[lon_idx]

    df_pixels = pd.DataFrame({
        "lat": grid_assigned_lats,
        "lon": grid_assigned_lons,
        "val": p_vals
    })

    # Group by lat, lon to compute spatial aggregations
    agg_df = df_pixels.groupby(["lat", "lon"])["val"].agg(
        val_mean="mean",
        val_min="min",
        val_max="max",
        val_std="std"
    ).reset_index()

    # Build full grid to ensure all points exist
    full_grid = []
    for lat in grid_lats:
        for lon in grid_lons:
            grid_id = f"G_{int(round(lat * 50))}_{int(round(lon * 50))}"
            full_grid.append({
                "timestamp": target_dt,
                "grid_id": grid_id,
                "lat": float(round(lat, 2)),
                "lon": float(round(lon, 2))
            })
    df_full = pd.DataFrame(full_grid)

    merged = df_full.merge(agg_df, on=["lat", "lon"], how="left")

    if dataset_name == "B13":
        merged["bt_mean"] = merged["val_mean"].round(2)
        merged["bt_min"] = merged["val_min"].round(2)
        merged["bt_max"] = merged["val_max"].round(2)
        merged["bt_std"] = merged["val_std"].fillna(0.0).round(2)
        cols = ["timestamp", "grid_id", "lat", "lon", "bt_mean", "bt_min", "bt_max", "bt_std"]
    else:
        merged["wv_mean"] = merged["val_mean"].round(2)
        cols = ["timestamp", "grid_id", "lat", "lon", "wv_mean"]

    return merged[cols]


def fetch_himawari_observation_aws(
    target_dt: pd.Timestamp,
    band: str = "band13",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cleanup_dat: bool = True
) -> pd.DataFrame:
    """
    High-level API: Fetches, extracts, and produces real Himawari observation DataFrame.
    """
    bucket, keys = build_s3_prefix_and_keys(target_dt, band=band)
    logger.info("Fetching real Himawari %s from AWS S3 (%s)...", band, bucket)

    dat_files = download_and_extract_hsd_segments(bucket, keys, cache_dir=cache_dir)
    try:
        df_grid = extract_satellite_band_to_grid(dat_files, target_dt, band=band)
    finally:
        if cleanup_dat:
            for p in dat_files:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    return df_grid


def check_coverage_across_years(
    years: List[int] = [2021, 2022, 2023, 2024, 2025]
) -> Dict[int, Dict[str, bool]]:
    """
    Checks data availability in AWS S3 for the specified years.
    Returns status mapping for each year.
    """
    s3_client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    summary = {}

    for y in years:
        # Check June 1 and Dec 1
        dts = [pd.Timestamp(f"{y}-06-01 00:00:00"), pd.Timestamp(f"{y}-12-01 00:00:00")]
        y_ok = True
        details = {}
        for dt in dts:
            bucket, keys = build_s3_prefix_and_keys(dt, band="B13", segments=[4])
            prefix = keys[0]
            resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            found = len(resp.get("Contents", [])) > 0
            tag = dt.strftime("%b")
            details[f"{tag}_{bucket}"] = found
            if not found:
                y_ok = False
        summary[y] = {"available": y_ok, "details": details}

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Checking AWS S3 Himawari Coverage for 2021-2025:")
    cov = check_coverage_across_years()
    for yr, res in cov.items():
        print(f"  Year {yr}: {'AVAILABLE' if res['available'] else 'MISSING'} -> {res['details']}")
