"""
TIGGE NWP Forecasts Extractor & Grid Transformer
Handles:
- Ingestion of raw GRIB2 or intermediate NWP forecast cycles
- Unit conversions (tp to mm, sp to hPa, 2t to Celsius)
- Forecast lead-time calculation (lead_time = valid_time - run_time)
- Spatial mapping to Thailand Grid
- Output to Analysis-Ready Parquet:
  data/nwp_forecasts/{origin}/run_{YYYYMMDD}_{HHz}.parquet
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_RAW_NWP_DIR = PROJECT_ROOT / "data" / "raw" / "nwp_runs"
DEFAULT_OUT_NWP_DIR = PROJECT_ROOT / "data" / "nwp_forecasts"


def normalize_nwp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes units and validates meteorological sanity bounds:
    - tp (total precipitation): mm (if in meters, * 1000)
    - sp (surface pressure): hPa (if in Pa, / 100)
    - t2m (2m temperature): Celsius (if in Kelvin, - 273.15)
    - u10, v10: m/s
    - cape: J/kg
    """
    clean_df = df.copy()

    # Total precipitation check (if max < 1.0 and unit was meters, convert to mm)
    if "tp" in clean_df.columns:
        if clean_df["tp"].max() > 0 and clean_df["tp"].max() < 0.5:
            clean_df["tp"] = clean_df["tp"] * 1000.0
        clean_df["tp"] = clean_df["tp"].clip(lower=0.0, upper=300.0)

    # Surface pressure check (if > 2000, it's Pa -> convert to hPa)
    if "sp" in clean_df.columns:
        if clean_df["sp"].mean() > 2000.0:
            clean_df["sp"] = clean_df["sp"] / 100.0
        clean_df["sp"] = clean_df["sp"].clip(lower=800.0, upper=1050.0)

    # Temperature check (if > 200, it's Kelvin -> convert to Celsius)
    temp_col = "t2m" if "t2m" in clean_df.columns else "2t"
    if temp_col in clean_df.columns:
        if clean_df[temp_col].mean() > 200.0:
            clean_df["t2m"] = clean_df[temp_col] - 273.15
        else:
            clean_df["t2m"] = clean_df[temp_col]
        clean_df["t2m"] = clean_df["t2m"].clip(lower=-10.0, upper=50.0)

    # Calculate and verify lead time
    if "run_time" in clean_df.columns and "valid_time" in clean_df.columns:
        clean_df["run_time"] = pd.to_datetime(clean_df["run_time"])
        clean_df["valid_time"] = pd.to_datetime(clean_df["valid_time"])
        diff_hours = (clean_df["valid_time"] - clean_df["run_time"]).dt.total_seconds() / 3600.0
        clean_df["lead_time_hours"] = diff_hours.astype(int)

        # Invariant check: valid_time must always be strictly greater than or equal to run_time
        if (clean_df["valid_time"] < clean_df["run_time"]).any():
            raise ValueError("Found invalid forecast rows where valid_time < run_time (Causal Violation)!")

    # Grid ID calculation based on 2 km grid resolution (approx 0.018 degrees)
    # cell_x = round(lon / 0.02), cell_y = round(lat / 0.02)
    clean_df["grid_id"] = (
        "G_" + (clean_df["lat"] * 50).round().astype(int).astype(str) +
        "_" + (clean_df["lon"] * 50).round().astype(int).astype(str)
    )

    columns_order = [
        "origin", "run_time", "valid_time", "lead_time_hours",
        "grid_id", "lat", "lon", "tp", "sp", "t2m", "u10", "v10", "cape"
    ]
    present_cols = [c for c in columns_order if c in clean_df.columns]
    return clean_df[present_cols]


def process_raw_nwp_file(input_file: Path, output_base_dir: Path = DEFAULT_OUT_NWP_DIR) -> Optional[Path]:
    """Processes a single raw NWP file (GRIB2 or intermediate Parquet) into Analysis-Ready Parquet."""
    logger.info("Extracting NWP data from %s...", input_file.name)
    try:
        if input_file.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(input_file)
        elif input_file.suffix in (".grib", ".grib2", ".grb"):
            import xarray as xr
            ds = xr.open_dataset(input_file, engine="cfgrib")
            df = ds.to_dataframe().reset_index()
        else:
            logger.warning("Unsupported file format: %s", input_file)
            return None

        clean_df = normalize_nwp_dataframe(df)

        # Extract origin and run time from filename or data
        origin = clean_df["origin"].iloc[0] if "origin" in clean_df.columns else "unknown"
        run_dt = clean_df["run_time"].iloc[0]
        run_tag = run_dt.strftime("%Y%m%d_%Hz").lower()

        target_dir = output_base_dir / origin
        target_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = target_dir / f"run_{run_tag}.parquet"

        clean_df.to_parquet(out_parquet, index=False, compression="snappy")
        logger.info("Saved Analysis-Ready NWP Parquet: %s (%d records)", out_parquet, len(clean_df))
        return out_parquet
    except Exception as e:
        logger.error("Failed processing %s: %s", input_file, e)
        return None


def run_extraction_pipeline(raw_dir: Path = DEFAULT_RAW_NWP_DIR, out_dir: Path = DEFAULT_OUT_NWP_DIR) -> List[Path]:
    """Scans raw NWP directory and transforms all available runs into Analysis-Ready Parquet."""
    files = list(raw_dir.glob("**/*.parquet")) + list(raw_dir.glob("**/*.grib*"))
    logger.info("Found %d raw NWP runs in %s", len(files), raw_dir)
    results = []
    for f in files:
        res = process_raw_nwp_file(f, output_base_dir=out_dir)
        if res:
            results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser(description="TIGGE NWP Analysis-Ready Extractor")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_NWP_DIR), help="Input raw NWP dir")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_NWP_DIR), help="Output parquet dir")
    args = parser.parse_args()

    run_extraction_pipeline(raw_dir=Path(args.raw_dir), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
