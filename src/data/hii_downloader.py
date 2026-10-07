"""
HII Ground Weather Data Acquisition & Clean Parquet Converter
Converts raw HII Open Data streams into harmonized, zero-null verified Parquet partitions:
data/clean_parquet/{variable_type}/year={year}/{station_code}.parquet

Schema:
- station_code: string
- observed_at: timestamp[us]
- variable_type: string
- value: float32
- quality_flag: string ("VALID", "MISSING_TIMESTAMP", "SENTINEL_ERROR", etc.)
- is_golden: boolean
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.hii_parser import detect_and_normalize_csv, audit_station_records
from src.config.paths import (
    PROJECT_ROOT,
    DEFAULT_HII_CLEAN_DIR,
    DEFAULT_HII_AUDIT_DIR,
    resolve_hii_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLEAN_DIR = DEFAULT_HII_CLEAN_DIR
DEFAULT_AUDIT_DIR = DEFAULT_HII_AUDIT_DIR

CATALOG_BASE_URLS = {
    "hourly_rain": "https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain",
    "pressure": "https://tiservice.hii.or.th/opendata/data_catalog/pressure",
    "humidity": "https://tiservice.hii.or.th/opendata/data_catalog/humidity",
}

ALL_60_MONTHS = [f"{y}{m:02d}" for y in range(2021, 2026) for m in range(1, 13)]

PARQUET_SCHEMA = pa.schema([
    ("station_code", pa.string()),
    ("observed_at", pa.timestamp("us")),
    ("variable_type", pa.string()),
    ("value", pa.float32()),
    ("quality_flag", pa.string()),
    ("is_golden", pa.bool_()),
])


def fetch_csv_bytes(catalog: str, ym: str, station_code: str) -> Optional[bytes]:
    """Downloads one station CSV for a specific month."""
    year = ym[:4]
    url = f"{CATALOG_BASE_URLS[catalog]}/{year}/{ym}/{station_code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    except Exception:
        return None


def process_station_year(
    catalog: str,
    station_code: str,
    year: int,
    output_base_dir: Path
) -> Optional[Path]:
    """
    Downloads all months for a station in a given year,
    harmonizes, audits, and writes to partitioned Parquet file.
    """
    months = [f"{year}{m:02d}" for m in range(1, 13)]
    start_dt = pd.Timestamp(f"{year}-01-01 00:00:00")
    end_dt = pd.Timestamp(f"{year}-12-31 23:00:00")

    dfs = []
    for ym in months:
        raw_bytes = fetch_csv_bytes(catalog, ym, station_code)
        if raw_bytes:
            try:
                df_month = detect_and_normalize_csv(raw_bytes, station_code, catalog)
                if not df_month.empty:
                    dfs.append(df_month)
            except Exception as e:
                logger.warning("Error parsing %s %s %s: %s", catalog, ym, station_code, e)

    if not dfs:
        logger.warning("No data found for %s %s in year %d", catalog, station_code, year)
        return None

    merged_df = pd.concat(dfs, ignore_index=True)
    clean_df, metrics = audit_station_records(merged_df, start_dt, end_dt, catalog, station_code)

    if clean_df.empty:
        return None

    # Cast to schema types
    clean_df["station_code"] = clean_df["station_code"].astype(str)
    clean_df["observed_at"] = pd.to_datetime(clean_df["observed_at"])
    clean_df["variable_type"] = clean_df["variable_type"].astype(str)
    clean_df["value"] = clean_df["value"].astype("float32")
    clean_df["quality_flag"] = clean_df["quality_flag"].astype(str)
    clean_df["is_golden"] = clean_df["is_golden"].astype(bool)

    cols_order = ["station_code", "observed_at", "variable_type", "value", "quality_flag", "is_golden"]
    clean_df = clean_df[cols_order]

    # Partition path: data/clean_parquet/{catalog}/year={year}/{station_code}.parquet
    target_dir = output_base_dir / catalog / f"year={year}"
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = target_dir / f"{station_code}.parquet"

    table = pa.Table.from_pandas(clean_df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, parquet_file, compression="snappy")
    logger.info("Saved %s Parquet: %s (valid: %d/%d, %.2f%%)", catalog, parquet_file.name, metrics["valid_hours"], metrics["total_expected_hours"], metrics["completeness_pct"])
    return parquet_file


def download_clean_data(
    stations: List[str],
    years: List[int],
    catalogs: List[str],
    output_dir: Path = DEFAULT_CLEAN_DIR,
    max_workers: int = 4
):
    """Orchestrates batch downloading and Parquet conversion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting acquisition & conversion: %d stations, %s years, %s catalogs", len(stations), years, catalogs)

    total_tasks = []
    for cat in catalogs:
        for stn in stations:
            for yr in years:
                total_tasks.append((cat, stn, yr))

    total_count = len(total_tasks)
    logger.info("Total conversion tasks: %d", total_count)
    saved_files = []
    completed_count = 0

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_task = {
            ex.submit(process_station_year, c, s, y, output_dir): (c, s, y)
            for c, s, y in total_tasks
        }
        for future in as_completed(future_to_task):
            completed_count += 1
            c, s, y = future_to_task[future]
            pct = (completed_count / total_count) * 100
            try:
                res = future.result()
                if res is not None:
                    saved_files.append(res)
                    logger.info("[Progress: %d/%d (%.1f%%)] Done %s | Station: %s | Year: %d -> %s",
                                completed_count, total_count, pct, c, s, y, res.name)
                else:
                    logger.warning("[Progress: %d/%d (%.1f%%)] No data for %s | Station: %s | Year: %d",
                                   completed_count, total_count, pct, c, s, y)
            except Exception as e:
                logger.error("[Progress: %d/%d (%.1f%%)] Failed %s | Station: %s | Year: %d: %s",
                             completed_count, total_count, pct, c, s, y, e)

    logger.info("Acquisition complete! Successfully written %d/%d Parquet partitions to %s",
                len(saved_files), total_count, output_dir)
    return saved_files


def get_candidate_stations(sample_count: int = 5) -> List[str]:
    """Finds top candidates from golden report or defaults to well-known stations."""
    golden_csv = DEFAULT_AUDIT_DIR / "golden_stations_zero_null_2021_2025.csv"
    if golden_csv.exists():
        try:
            df = pd.read_csv(golden_csv)
            if not df.empty and "station_code" in df.columns:
                stns = df["station_code"].unique().tolist()
                return stns[:sample_count]
        except Exception:
            pass

    # Fallback to stations with known high reliability from catalog
    return ["ABRT", "ACRU", "AIT002", "AIT003", "AIT004"][:sample_count]


def main():
    parser = argparse.ArgumentParser(description="HII Clean Parquet Downloader")
    parser.add_argument("--hii-dir", type=str, default=None, help="Base directory for HII data (e.g. D:/data/hii)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output parquet directory (default: {hii-dir}/clean_parquet or data/clean_parquet)")
    parser.add_argument("--sample", action="store_true", help="Dev sample mode (limited stations)")
    parser.add_argument("--full", action="store_true", help="Production full mode")
    parser.add_argument("--stations", nargs="+", help="Station code(s) or number of stations")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2025], help="Years to acquire (e.g. 2021 2025 or 2021 2022 2023 2024 2025)")
    parser.add_argument("--all-catalogs", action="store_true", help="Download all 3 catalogs")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrency workers")
    args = parser.parse_args()

    # Determine paths via centralized resolver
    paths = resolve_hii_paths(hii_dir=args.hii_dir, clean_dir=args.output_dir)
    output_dir = paths["clean_dir"]

    # Determine stations
    if args.stations:
        if len(args.stations) == 1 and args.stations[0].isdigit():
            count = int(args.stations[0])
            stations = get_candidate_stations(count)
        else:
            stations = args.stations
    else:
        sample_count = 5 if args.sample or not args.full else 50
        stations = get_candidate_stations(sample_count)

    catalogs = ["hourly_rain", "pressure", "humidity"]
    years = args.years if args.years else [2021, 2025]

    download_clean_data(
        stations=stations,
        years=years,
        catalogs=catalogs,
        output_dir=output_dir,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
