"""
HII Station Metadata Acquisition & Cross-Mapping Module
Fetches station metadata catalogs from HII Open Data and produces:
- data/metadata/hii_stations_master_metadata.csv
- data/metadata/hii_stations_cross_mapping.json
"""

import os
import sys
import io
import json
import logging
import urllib.request
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.config.paths import DEFAULT_HII_META_DIR, resolve_hii_paths

CATALOG_URLS = {
    "hourly_rain": "https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain/0all_stn_metadata.csv",
    "pressure": "https://tiservice.hii.or.th/opendata/data_catalog/pressure/0all_stn_metadata.csv",
    "humidity": "https://tiservice.hii.or.th/opendata/data_catalog/humidity/0all_stn_metadata.csv",
}

DEFAULT_OUTPUT_DIR = DEFAULT_HII_META_DIR


def fetch_metadata_df(catalog_name: str, url: str) -> pd.DataFrame:
    """Download and parse 0all_stn_metadata.csv for a catalog."""
    logger.info("Fetching metadata for %s from %s", catalog_name, url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    # Try utf-8 first, fallback to tis-620
    df = None
    for enc in ["utf-8-sig", "utf-8", "tis-620", "cp874"]:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Failed to decode metadata CSV for {catalog_name}")

    # Standardize column names to lowercase snake_case
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["catalog"] = catalog_name
    return df


def build_master_metadata(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """
    Downloads metadata from all 3 catalogs, merges into a unified master metadata CSV,
    and creates cross-mapping JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    catalog_dfs = {}
    catalog_stations = {}
    
    for name, url in CATALOG_URLS.items():
        df = fetch_metadata_df(name, url)
        catalog_dfs[name] = df
        catalog_stations[name] = set(df["station_code"].unique())
        logger.info("Catalog '%s': %d unique stations found", name, len(catalog_stations[name]))

    # Rain catalog serves as the primary base
    base_df = catalog_dfs["hourly_rain"].copy()
    all_station_codes = set().union(*catalog_stations.values())
    
    logger.info("Total unique station codes across all catalogs: %d", len(all_station_codes))

    # Add flags for catalog availability
    master_df = base_df.drop_duplicates(subset=["station_code"]).copy()
    master_df["has_rain"] = master_df["station_code"].isin(catalog_stations["hourly_rain"])
    master_df["has_pressure"] = master_df["station_code"].isin(catalog_stations["pressure"])
    master_df["has_humidity"] = master_df["station_code"].isin(catalog_stations["humidity"])
    master_df["has_all_three"] = master_df["has_rain"] & master_df["has_pressure"] & master_df["has_humidity"]

    # Filter/Clean Coordinates (Thailand bounding box roughly: lat 5 to 21, lon 97 to 106)
    master_df["latitude"] = pd.to_numeric(master_df["latitude"], errors="coerce")
    master_df["longitude"] = pd.to_numeric(master_df["longitude"], errors="coerce")
    valid_coords = (
        (master_df["latitude"] >= 5.0) & (master_df["latitude"] <= 21.5) &
        (master_df["longitude"] >= 96.5) & (master_df["longitude"] <= 106.5)
    )
    master_df["valid_thailand_coords"] = valid_coords

    # Save master metadata CSV
    master_csv_path = output_dir / "hii_stations_master_metadata.csv"
    master_df.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved master metadata to %s (%d records)", master_csv_path, len(master_df))

    # Build cross mapping JSON
    cross_mapping = {
        "summary": {
            "total_stations": len(master_df),
            "rain_stations_count": len(catalog_stations["hourly_rain"]),
            "pressure_stations_count": len(catalog_stations["pressure"]),
            "humidity_stations_count": len(catalog_stations["humidity"]),
            "stations_with_all_three_catalogs": int(master_df["has_all_three"].sum()),
            "stations_with_valid_coords": int(master_df["valid_thailand_coords"].sum()),
        },
        "stations": {}
    }

    for _, row in master_df.iterrows():
        code = row["station_code"]
        cross_mapping["stations"][code] = {
            "station_code": code,
            "station_name": row.get("station_name", ""),
            "latitude": float(row["latitude"]) if pd.notnull(row["latitude"]) else None,
            "longitude": float(row["longitude"]) if pd.notnull(row["longitude"]) else None,
            "basin_name": row.get("basin_name", ""),
            "sub_basin_name": row.get("sub_basin_name", ""),
            "province_name": row.get("province_name", ""),
            "amphoe_name": row.get("amphoe_name", ""),
            "tambon_name": row.get("tambon_name", ""),
            "has_rain": bool(row["has_rain"]),
            "has_pressure": bool(row["has_pressure"]),
            "has_humidity": bool(row["has_humidity"]),
            "has_all_three": bool(row["has_all_three"]),
        }

    cross_mapping_path = output_dir / "hii_stations_cross_mapping.json"
    with open(cross_mapping_path, "w", encoding="utf-8") as f:
        json.dump(cross_mapping, f, ensure_ascii=False, indent=2)
    logger.info("Saved cross mapping to %s", cross_mapping_path)

    return cross_mapping["summary"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HII Station Metadata Acquisition & Cross-Mapping")
    parser.add_argument("--hii-dir", type=str, default=None, help="Base directory for HII data (e.g. D:/data/hii)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output metadata directory (default: {hii-dir}/metadata or data/metadata)")
    args = parser.parse_args()

    paths = resolve_hii_paths(hii_dir=args.hii_dir, meta_dir=args.output_dir)
    target_out = paths["meta_dir"]

    summary = build_master_metadata(output_dir=target_out)
    print("Metadata generation complete:")
    print(json.dumps(summary, indent=2))
