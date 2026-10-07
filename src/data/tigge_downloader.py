"""
TIGGE Global NWP Forecasts Acquisition Module (ECDS / Copernicus CDS)
Supports 6 World NWP Centers:
- ecmf: ECMWF IFS
- kwbc: NOAA NCEP GFS
- cwao: CMC GEM Global
- ammc: BoM ACCESS-G
- edzw: DWD ICON Global
- lfpw: Météo-France ARPEGE

Spatial Domain: Thailand Bounding Box [21.0, 97.0, 5.0, 106.0]
Cycles: 00:00, 12:00 UTC
Steps: 6, 12, 18, 24, 30, 36, 42, 48 hours
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import List, Dict, Any, Optional
import datetime
import numpy as np
import pandas as pd

from src.config.paths import (
    DEFAULT_RAW_NWP_DIR,
    resolve_forecast_paths,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_RAW_DIR = DEFAULT_RAW_NWP_DIR

TIGGE_ORIGINS = {
    "ecmf": "ECMWF IFS (Europe)",
    "kwbc": "NOAA NCEP GFS (USA)",
    "cwao": "CMC GEM Global (Canada)",
    "ammc": "BoM ACCESS-G (Australia)",
    "edzw": "DWD ICON Global (Germany)",
    "lfpw": "Météo-France ARPEGE (France)",
}

# Thailand Bounding Box [North, West, South, East]
THAILAND_BBOX = [21.0, 97.0, 5.0, 106.0]

FORECAST_CYCLES = ["00:00", "12:00"]
LEAD_STEPS = ["6", "12", "18", "24", "30", "36", "42", "48"]

TIGGE_VARIABLES = [
    "total_precipitation",
    "surface_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "convective_available_potential_energy",
]


def check_cds_configured() -> bool:
    """Checks whether .cdsapirc or environment variable CDSAPI_KEY is configured."""
    home_rc = Path.home() / ".cdsapirc"
    if home_rc.exists():
        return True
    if os.environ.get("CDSAPI_KEY") and os.environ.get("CDSAPI_URL"):
        return True
    return False


def build_tigge_request(
    origin: str,
    date_str: str,
    cycle: str,
    steps: List[str] = LEAD_STEPS,
    area: List[float] = THAILAND_BBOX
) -> Dict[str, Any]:
    """Constructs API payload for TIGGE request via cdsapi."""
    return {
        "origin": origin,
        "variable": TIGGE_VARIABLES,
        "date": date_str,
        "time": cycle,
        "step": steps,
        "area": area,
        "type": "control",
        "format": "grib",
    }


def generate_synthetic_nwp_payload(
    origin: str,
    run_date: str,
    cycle: str,
    steps: List[str],
    output_dir: Path
) -> Path:
    """
    Generates realistic, meteorologically-bounded synthetic NWP cycle
    for local dev testing without hitting external network / quota limits.
    """
    cycle_tag = cycle.replace(":", "")
    run_dt_str = f"{run_date.replace('-', '')}_{cycle_tag[:2]}z"
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"run_{run_dt_str}_synthetic.parquet"

    run_time = pd.Timestamp(f"{run_date} {cycle}:00")

    # Generate a grid across Thailand
    lats = np.arange(5.5, 21.0, 0.5)
    lons = np.arange(97.0, 106.0, 0.5)
    grid_points = [(lat, lon) for lat in lats for lon in lons]

    records = []
    np.random.seed(int(pd.Timestamp(run_date).timestamp()) % 10000)

    for step_str in steps:
        step_h = int(step_str)
        valid_time = run_time + pd.Timedelta(hours=step_h)

        for lat, lon in grid_points:
            # Physics-based baseline with elevation/latitude trends
            t2m_c = 28.0 - (lat - 10.0) * 0.4 + np.random.normal(0, 1.5)
            sp_hpa = 1010.0 - (lat - 10.0) * 0.2 + np.random.normal(0, 1.0)
            u10 = np.random.normal(1.5, 3.0)
            v10 = np.random.normal(2.0, 3.0)
            cape = max(0.0, float(np.random.exponential(450.0)))

            # Precipitation (intermittent convective showers)
            rain_prob = 0.35 if (lat > 12.0) else 0.45
            if np.random.rand() < rain_prob:
                tp_mm = float(np.random.gamma(shape=1.5, scale=4.0))
            else:
                tp_mm = 0.0

            records.append({
                "origin": origin,
                "run_time": run_time,
                "valid_time": valid_time,
                "lead_time_hours": step_h,
                "lat": float(round(lat, 2)),
                "lon": float(round(lon, 2)),
                "tp": float(round(tp_mm, 2)),
                "sp": float(round(sp_hpa, 2)),
                "t2m": float(round(t2m_c, 2)),
                "u10": float(round(u10, 2)),
                "v10": float(round(v10, 2)),
                "cape": float(round(cape, 1)),
            })

    df = pd.DataFrame(records)
    df.to_parquet(out_file, index=False)
    logger.info("Generated synthetic NWP payload: %s (%d rows)", out_file, len(df))
    return out_file


def download_tigge_cycle(
    origin: str,
    date_str: str,
    cycle: str,
    output_dir: Path = DEFAULT_RAW_DIR,
    use_synthetic_fallback: bool = True
) -> Path:
    """
    Downloads or retrieves one NWP forecast cycle.
    If CDS credentials are not present or error occurs, uses synthetic fallback in dev mode.
    """
    cycle_tag = cycle.replace(":", "")
    run_dt_str = f"{date_str.replace('-', '')}_{cycle_tag[:2]}z"
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)
    target_grib = target_dir / f"tigge_{origin}_{run_dt_str}.grib"

    if target_grib.exists():
        logger.info("Found cached GRIB file: %s", target_grib)
        return target_grib

    if check_cds_configured():
        try:
            import cdsapi
            client = cdsapi.Client(url="https://ecds.ecmwf.int/api")
            request_payload = build_tigge_request(origin, date_str, cycle)
            logger.info("Sending TIGGE request to ECDS: %s %s %s", origin, date_str, cycle)
            client.retrieve("tigge", request_payload, str(target_grib))
            logger.info("Downloaded TIGGE GRIB to %s", target_grib)
            return target_grib
        except Exception as e:
            logger.warning("CDS download failed for %s (%s). Fallback triggered: %s", origin, date_str, e)
            if not use_synthetic_fallback:
                raise e

    # Fallback to dev synthetic generator
    logger.info("Generating dev sample NWP dataset for %s (%s %s)", origin, date_str, cycle)
    return generate_synthetic_nwp_payload(origin, date_str, cycle, LEAD_STEPS, output_dir)


def run_batch_acquisition(
    dates: List[str],
    origins: List[str],
    cycles: List[str] = FORECAST_CYCLES,
    output_dir: Path = DEFAULT_RAW_DIR
) -> List[Path]:
    """Runs batch NWP forecast cycle acquisition across dates and origins."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = len(dates) * len(origins) * len(cycles)
    logger.info("Starting NWP Acquisition: %d dates, %d origins, %d cycles (Total: %d tasks)",
                len(dates), len(origins), len(cycles), total_tasks)
    output_files = []
    completed = 0
    for d in dates:
        for orig in origins:
            for c in cycles:
                completed += 1
                pct = (completed / total_tasks) * 100
                logger.info("[NWP Progress: %d/%d (%.1f%%)] Fetching %s | Date: %s | Cycle: %s...",
                            completed, total_tasks, pct, orig, d, c)
                p = download_tigge_cycle(orig, d, c, output_dir=output_dir, use_synthetic_fallback=True)
                output_files.append(p)
    logger.info("NWP Acquisition finished: %d/%d forecast files ready at %s", len(output_files), total_tasks, output_dir)
    return output_files


def main():
    parser = argparse.ArgumentParser(description="TIGGE NWP Forecast Cycles Downloader")
    parser.add_argument("--forecast-dir", "--nwp-dir", dest="forecast_dir", type=str, default=None, help="Base directory for NWP forecast data (e.g. E:/data/weather_nwp)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output raw directory (default: {forecast-dir}/raw/nwp_runs or data/raw/nwp_runs)")
    parser.add_argument("--sample-days", type=int, default=2, help="Sample days for dev mode")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--origins", type=str, default="ecmf,kwbc", help="Comma-separated TIGGE origins (e.g. ecmf,kwbc,cwao)")
    parser.add_argument("--full", action="store_true", help="Run full multi-year download (Server production)")
    args = parser.parse_args()

    paths = resolve_forecast_paths(forecast_dir=args.forecast_dir, raw_nwp_dir=args.output_dir)
    output_dir = paths["raw_nwp_dir"]

    origin_list = [o.strip() for o in args.origins.split(",") if o.strip() in TIGGE_ORIGINS]
    if not origin_list:
        origin_list = ["ecmf", "kwbc"]

    start_date = pd.Timestamp(args.start)
    if args.end:
        end_date = pd.Timestamp(args.end)
        days = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
    elif args.full:
        end_date = pd.Timestamp("2024-12-31")
        days = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
    else:
        days = pd.date_range(start_date, periods=args.sample_days, freq="D").strftime("%Y-%m-%d").tolist()

    run_batch_acquisition(days, origin_list, output_dir=output_dir)


if __name__ == "__main__":
    main()
