"""
Centralized Storage Path Configuration for weather-forcast-enhance
Supports configuring HII Ground Data and Weather Forecast Data independently
via CLI Arguments, Environment Variables, or Default Paths.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

# Workspace / Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Base Default Data Directory
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# --------------------------------------------------------------------------
# 1. HII Ground Station Data Defaults (Can be pointed to separate storage/drive)
# --------------------------------------------------------------------------
DEFAULT_HII_DIR = Path(os.getenv("HII_DATA_DIR", DEFAULT_DATA_DIR))
DEFAULT_HII_CLEAN_DIR = Path(os.getenv("HII_CLEAN_DIR", DEFAULT_HII_DIR / "clean_parquet"))
DEFAULT_HII_META_DIR = Path(os.getenv("HII_META_DIR", DEFAULT_HII_DIR / "metadata"))
DEFAULT_HII_AUDIT_DIR = Path(os.getenv("HII_AUDIT_DIR", DEFAULT_HII_DIR / "audit"))

# --------------------------------------------------------------------------
# 2. Weather Forecast & Satellite Data Defaults (NWP GRIB2/Parquet, Himawari-9)
# --------------------------------------------------------------------------
DEFAULT_FORECAST_DIR = Path(os.getenv("FORECAST_DATA_DIR", DEFAULT_DATA_DIR))
DEFAULT_RAW_NWP_DIR = Path(os.getenv("NWP_RAW_DIR", DEFAULT_FORECAST_DIR / "raw" / "nwp_runs"))
DEFAULT_OUT_NWP_DIR = Path(os.getenv("NWP_FORECASTS_DIR", DEFAULT_FORECAST_DIR / "nwp_forecasts"))
DEFAULT_HIMAWARI_DIR = Path(os.getenv("HIMAWARI_DIR", DEFAULT_FORECAST_DIR / "himawari9"))

# --------------------------------------------------------------------------
# 3. Common Geographic & Feature Matrix Defaults
# --------------------------------------------------------------------------
DEFAULT_GEO_DIR = Path(os.getenv("GEO_DATA_DIR", DEFAULT_DATA_DIR / "geo"))
DEFAULT_FEATURES_DIR = Path(os.getenv("FEATURES_DIR", DEFAULT_DATA_DIR / "features"))


def resolve_hii_paths(
    hii_dir: Optional[str] = None,
    clean_dir: Optional[str] = None,
    meta_dir: Optional[str] = None,
    audit_dir: Optional[str] = None
) -> Dict[str, Path]:
    """
    Resolves HII directory paths given optional CLI arguments.
    If `hii_dir` is provided, subdirectories default to `{hii_dir}/clean_parquet`, etc.
    Otherwise fallbacks to environment variables or project defaults.
    """
    base = Path(hii_dir) if hii_dir else DEFAULT_HII_DIR
    clean = Path(clean_dir) if clean_dir else (base / "clean_parquet" if hii_dir else DEFAULT_HII_CLEAN_DIR)
    meta = Path(meta_dir) if meta_dir else (base / "metadata" if hii_dir else DEFAULT_HII_META_DIR)
    audit = Path(audit_dir) if audit_dir else (base / "audit" if hii_dir else DEFAULT_HII_AUDIT_DIR)

    return {
        "hii_dir": base,
        "clean_dir": clean,
        "meta_dir": meta,
        "audit_dir": audit,
    }


def resolve_forecast_paths(
    forecast_dir: Optional[str] = None,
    raw_nwp_dir: Optional[str] = None,
    out_nwp_dir: Optional[str] = None,
    himawari_dir: Optional[str] = None
) -> Dict[str, Path]:
    """
    Resolves Weather Forecast directory paths given optional CLI arguments.
    If `forecast_dir` is provided, subdirectories default to `{forecast_dir}/raw/nwp_runs`, etc.
    Otherwise fallbacks to environment variables or project defaults.
    """
    base = Path(forecast_dir) if forecast_dir else DEFAULT_FORECAST_DIR
    raw = Path(raw_nwp_dir) if raw_nwp_dir else (base / "raw" / "nwp_runs" if forecast_dir else DEFAULT_RAW_NWP_DIR)
    out = Path(out_nwp_dir) if out_nwp_dir else (base / "nwp_forecasts" if forecast_dir else DEFAULT_OUT_NWP_DIR)
    sat = Path(himawari_dir) if himawari_dir else (base / "himawari9" if forecast_dir else DEFAULT_HIMAWARI_DIR)

    return {
        "forecast_dir": base,
        "raw_nwp_dir": raw,
        "out_nwp_dir": out,
        "himawari_dir": sat,
    }
