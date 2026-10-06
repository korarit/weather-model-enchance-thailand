"""
HII Ground Data Acquisition & Zero-Null Completeness Audit Pipeline
Covers:
- Stage 1: Remote Presence Matrix Scan (2021-2025, 60 months)
- Stage 2: Deep Row-Level Stream Inspection & Zero-Null Tier Auditing
- Multi-Catalog Joint Intersection Analysis (Rain + Pressure + Humidity)
- Deliverable Generation: CSV reports, spatial map, and audit summary report
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import re
import argparse
import logging
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.hii_parser import detect_and_normalize_csv, audit_station_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
DEFAULT_META_DIR = PROJECT_ROOT / "data" / "metadata"

CATALOG_BASE_URLS = {
    "hourly_rain": "https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain",
    "pressure": "https://tiservice.hii.or.th/opendata/data_catalog/pressure",
    "humidity": "https://tiservice.hii.or.th/opendata/data_catalog/humidity",
}

ALL_60_MONTHS = [f"{y}{m:02d}" for y in range(2021, 2026) for m in range(1, 13)]


def fetch_month_station_list(catalog: str, ym: str) -> Tuple[str, Set[str]]:
    """Scans one monthly directory for available station CSV filenames."""
    year = ym[:4]
    url = f"{CATALOG_BASE_URLS[catalog]}/{year}/{ym}/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", "ignore")
        stns = set(f[:-4] for f in re.findall(r'href="([^"?]+\.csv)"', html) if not f.startswith("0"))
        return ym, stns
    except Exception as e:
        logger.warning("Failed scanning %s %s: %s", catalog, ym, e)
        return ym, set()


def scan_presence_matrix(catalog: str, months: List[str], max_workers: int = 8) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Stage 1: Generates the presence matrix (Station x Months) for a catalog.
    """
    logger.info("Stage 1: Scanning presence matrix for '%s' across %d months...", catalog, len(months))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_month_station_list, catalog, ym) for ym in months]
        results = [f.result() for f in futures]

    stn_to_months = defaultdict(set)
    for ym, stn_set in results:
        for s in stn_set:
            stn_to_months[s].add(ym)

    all_stations = sorted(stn_to_months.keys())
    rows = []
    for s in all_stations:
        mset = stn_to_months[s]
        row = {
            "station_code": s,
            "catalog": catalog,
            "months_present_count": len(mset),
            "presence_rate_pct": round((len(mset) / len(months)) * 100.0, 2),
        }
        for ym in months:
            row[f"m_{ym}"] = 1 if ym in mset else 0
        rows.append(row)

    df_matrix = pd.DataFrame(rows)
    return df_matrix, stn_to_months


def fetch_station_month_csv(catalog: str, ym: str, station_code: str) -> Optional[bytes]:
    """Downloads one station monthly CSV file."""
    year = ym[:4]
    url = f"{CATALOG_BASE_URLS[catalog]}/{year}/{ym}/{station_code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    except Exception:
        return None


def inspect_station_deep(
    catalog: str,
    station_code: str,
    months: List[str],
    max_workers: int = 4
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Stage 2: Downloads all monthly CSVs for a station, stitches them,
    and runs row-level audit against the full hourly timestamp range.
    """
    start_dt = pd.Timestamp(f"{months[0][:4]}-{months[0][4:6]}-01 00:00:00")
    # End dt is last day of the last month
    last_ym = months[-1]
    last_year = int(last_ym[:4])
    last_month = int(last_ym[4:6])
    next_month_dt = pd.Timestamp(f"{last_year}-{last_month:02d}-01") + pd.DateOffset(months=1)
    end_dt = next_month_dt - pd.Timedelta(hours=1)

    dfs = []
    def get_month(ym):
        content = fetch_station_month_csv(catalog, ym, station_code)
        if content:
            try:
                return detect_and_normalize_csv(content, station_code, catalog)
            except Exception:
                return None
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(get_month, months))

    for df in results:
        if df is not None and not df.empty:
            dfs.append(df)

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
    else:
        merged_df = pd.DataFrame()

    cleaned_df, metrics = audit_station_records(merged_df, start_dt, end_dt, catalog, station_code)
    metrics["catalog"] = catalog
    metrics["months_analyzed"] = len(months)
    return cleaned_df, metrics


def load_metadata() -> pd.DataFrame:
    """Loads station master metadata if available."""
    meta_path = DEFAULT_META_DIR / "hii_stations_master_metadata.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
    return pd.DataFrame()


def generate_spatial_map(
    audit_results_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_png_path: Path
):
    """Plots spatial distribution of stations by quality tier across Thailand."""
    logger.info("Plotting spatial distribution map...")
    if metadata_df.empty or audit_results_df.empty:
        logger.warning("Metadata or audit results empty, skipping spatial map generation")
        return

    merged = audit_results_df.copy()
    if "latitude" not in merged.columns or "longitude" not in merged.columns:
        merged = merged.merge(
            metadata_df[["station_code", "latitude", "longitude", "province_name", "basin_name"]].drop_duplicates("station_code"),
            on="station_code",
            how="left"
        )

    merged["latitude"] = pd.to_numeric(merged["latitude"], errors="coerce")
    merged["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce")
    merged = merged.dropna(subset=["latitude", "longitude"])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    tier_styles = {
        "Tier 1: Golden": {"color": "#F59E0B", "marker": "*", "size": 90, "label": "Tier 1: Golden Zero-Null (100%)", "zorder": 5},
        "Tier 2: Silver": {"color": "#38BDF8", "marker": "o", "size": 45, "label": "Tier 2: Silver (>= 99.5%)", "zorder": 4},
        "Tier 3: Bronze": {"color": "#A78BFA", "marker": "^", "size": 35, "label": "Tier 3: Bronze (95-99.4%)", "zorder": 3},
        "Tier 4: Excluded": {"color": "#64748B", "marker": "x", "size": 25, "label": "Tier 4: Excluded (< 95%)", "zorder": 2},
    }

    # Plot by tier
    for tier_name, style in tier_styles.items():
        subset = merged[merged["tier"] == tier_name]
        if not subset.empty:
            ax.scatter(
                subset["longitude"],
                subset["latitude"],
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                alpha=0.85,
                label=f"{style['label']} ({len(subset)})",
                zorder=style["zorder"],
                edgecolors="none" if style["marker"] == "x" else "black",
                linewidth=0.5
            )

    ax.set_title("HII Ground Weather Stations Completeness Audit (2021-2025)\nSpatial Distribution Across Thailand", fontsize=14, color="#F8FAFC", weight="bold", pad=15)
    ax.set_xlabel("Longitude (°E)", fontsize=11, color="#CBD5E1")
    ax.set_ylabel("Latitude (°N)", fontsize=11, color="#CBD5E1")
    ax.set_xlim(97.0, 106.0)
    ax.set_ylim(5.5, 20.8)
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#475569")

    legend = ax.legend(loc="lower left", frameon=True, facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)
    ax.grid(True, color="#334155", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved spatial distribution map to %s", output_png_path)


def write_summary_markdown(
    meta_summary: Dict,
    stage1_stats: Dict,
    stage2_df: pd.DataFrame,
    joint_df: pd.DataFrame,
    output_md_path: Path
):
    """Generates the executive and technical audit summary report."""
    logger.info("Writing audit summary report to %s", output_md_path)

    t1_count = int((stage2_df["tier"] == "Tier 1: Golden").sum()) if not stage2_df.empty else 0
    t2_count = int((stage2_df["tier"] == "Tier 2: Silver").sum()) if not stage2_df.empty else 0
    t3_count = int((stage2_df["tier"] == "Tier 3: Bronze").sum()) if not stage2_df.empty else 0
    t4_count = int((stage2_df["tier"] == "Tier 4: Excluded").sum()) if not stage2_df.empty else 0
    joint_count = len(joint_df) if not joint_df.empty else 0

    lines = [
        "# 📊 HII Ground Data Zero-Null Completeness Audit Report (2021–2025)",
        "",
        "> **Project**: Weather Model Precipitation Bias Correction (2 km × 2 km Grid)",
        "> **Catalog Scope**: Hourly Rain, Air Pressure, Relative Humidity",
        "> **Temporal Scope**: 2021-01-01 00:00:00 to 2025-12-31 23:00:00 (5 Years / 60 Months / 43,824 Hours)",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Catalog Master Stations**: {meta_summary.get('total_stations', 1396):,} stations across Thailand",
        f"- **Stage 1 Stations with 100% Monthly Continuity (60/60 Months)**: {stage1_stats.get('stations_60_months_rain', 'N/A')} stations",
        f"- **Stage 2 Inspected Stations Count**: {len(stage2_df):,} stations",
        f"- **Tier 1 (Golden Zero-Null 100% Completeness)**: `{t1_count}` station-catalog pairs",
        f"- **Tier 2 (Silver High-Quality >= 99.5%, gap <= 3h)**: `{t2_count}` station-catalog pairs",
        f"- **Tier 3 (Bronze Operational 95.0% - 99.4%)**: `{t3_count}` station-catalog pairs",
        f"- **Tier 4 (Excluded < 95.0%)**: `{t4_count}` station-catalog pairs",
        f"- **Joint Triple-Golden Stations (Rain + Pressure + Humidity simultaneously 100%)**: `{joint_count}` stations",
        "",
        "---",
        "",
        "## 2. Stage 1: Remote Presence Matrix Scan Results",
        "",
        "| Catalog Variable | Total Stations Detected | Stations with Full 60 Months (100%) | 60-Month Continuity Rate |",
        "|---|---|---|---|",
    ]

    for cat in ["hourly_rain", "pressure", "humidity"]:
        tot = stage1_stats.get(f"{cat}_total", 0)
        c60 = stage1_stats.get(f"{cat}_60m", 0)
        pct = round((c60 / tot * 100.0), 2) if tot > 0 else 0
        lines.append(f"| **{cat}** | {tot:,} | {c60:,} | {pct}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Station Quality Tier Breakdown (Stage 2 Deep Inspection)",
        "",
        "| Quality Tier | Criteria | Count | Usage Recommendation |",
        "|---|---|---|---|",
        f"| 🏆 **Tier 1: Golden** | 100% Valid (0 Null, 0 Sentinel, 0 Gap, 0 Flag Error) | **{t1_count}** | **Ground Truth Target** for Phase 5 Evaluation & Core Training |",
        f"| 🥈 **Tier 2: Silver** | Completeness >= 99.5%, Max Gap <= 3 hours | **{t2_count}** | Spatial Ground Observation with Spline Imputation (Flag: `INTERPOLATED`) |",
        f"| 🥉 **Tier 3: Bronze** | Completeness 95.0% – 99.4% | **{t3_count}** | Secondary spatial density features |",
        f"| ❌ **Tier 4: Excluded** | Completeness < 95.0% | **{t4_count}** | **Excluded** from training and evaluation |",
        "",
        "---",
        "",
        "## 4. Top Golden Stations (Sample)",
        "",
    ])

    if not stage2_df.empty:
        golden_df = stage2_df[stage2_df["is_golden"]].copy()
        if not golden_df.empty:
            cols = ["station_code", "catalog", "valid_hours", "total_expected_hours", "completeness_pct", "tier"]
            sample_g = golden_df[[c for c in cols if c in golden_df.columns]].head(15)
            lines.append(sample_g.to_markdown(index=False))
        else:
            lines.append("*(No stations achieved strict 100% zero-null across the inspected subset)*")
    else:
        lines.append("*(Inspection in progress)*")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Joint Triple-Station Intersection (Rain + Pressure + Humidity)",
        "",
        f"Stations that possess **complete records for all 3 meteorological variables** concurrently at the same physical sensor location: **{joint_count}** stations.",
        "",
    ])

    if not joint_df.empty:
        lines.append(joint_df.head(15).to_markdown(index=False))
    else:
        lines.append("*(Joint intersection table will be populated upon complete cross-catalog run)*")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Next Steps for Phase 2 & ML Pipeline",
        "",
        "1. **Clean Parquet Archiving**: Run `hii_downloader.py` to persist verified clean records partitioned by catalog and year in `data/clean_parquet/`.",
        "2. **Spatial Nearest Neighbor Matching**: For stations lacking co-located pressure/humidity sensors, use IDW (Inverse Distance Weighting, <= 5 km radius) to pair spatial pressure/humidity observations as specified in `plan.md`.",
        "3. **Proceed to Phase 2**: Download Himawari-9 AHI IR/VIS satellite bands and NWP models (ECMWF IFS, AIFS, NCEP GEFS).",
        "",
    ])

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Audit summary report saved to %s", output_md_path)


def run_audit(
    mode: str = "sample",
    limit_stations: int = 5,
    months: List[str] = None,
    catalogs: List[str] = None,
    output_dir: Path = DEFAULT_AUDIT_DIR,
    max_workers: int = 8
):
    """Main audit execution controller."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if catalogs is None:
        catalogs = ["hourly_rain", "pressure", "humidity"]
    if months is None:
        months = ALL_60_MONTHS

    logger.info("Starting HII Audit [mode=%s, catalogs=%s, months=%d]", mode, catalogs, len(months))

    # Load master metadata
    metadata_df = load_metadata()
    meta_summary = {
        "total_stations": len(metadata_df) if not metadata_df.empty else 1396
    }

    # Stage 1: Presence Matrix Scan
    stage1_stats = {}
    catalog_matrices = {}
    catalog_stn_months = {}

    for cat in catalogs:
        df_mat, stn_m = scan_presence_matrix(cat, months, max_workers=max_workers)
        catalog_matrices[cat] = df_mat
        catalog_stn_months[cat] = stn_m
        c60 = int((df_mat["months_present_count"] == len(months)).sum())
        stage1_stats[f"{cat}_total"] = len(df_mat)
        stage1_stats[f"{cat}_60m"] = c60
        logger.info("Catalog '%s': %d stations, %d have 100%% (%d/%d) months", cat, len(df_mat), c60, len(months), len(months))

    # Combine presence matrices
    all_mat_df = pd.concat(list(catalog_matrices.values()), ignore_index=True)
    mat_csv_path = output_dir / "all_stations_completeness_matrix.csv"
    all_mat_df.to_csv(mat_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved presence matrix to %s", mat_csv_path)

    # Stage 2: Deep Row-Level Stream Inspection
    # Identify stations to inspect
    if mode == "sample":
        # Pick limit_stations stations that have all months in rain catalog
        rain_mat = catalog_matrices.get("hourly_rain", all_mat_df)
        full_stns = rain_mat[rain_mat["months_present_count"] == len(months)]["station_code"].tolist()
        if not full_stns:
            full_stns = rain_mat["station_code"].tolist()
        target_stns = full_stns[:limit_stations]
        logger.info("Sample mode: Inspecting %d stations: %s", len(target_stns), target_stns)
    else:
        # Full mode: Inspect all stations with full presence (or all detected stations)
        rain_mat = catalog_matrices.get("hourly_rain", all_mat_df)
        target_stns = rain_mat[rain_mat["months_present_count"] == len(months)]["station_code"].tolist()
        logger.info("Full mode: Inspecting %d candidate stations with complete month continuity", len(target_stns))

    stage2_results = []
    for cat in catalogs:
        logger.info("Deep inspecting %d stations for '%s'...", len(target_stns), cat)
        for s in target_stns:
            _, metrics = inspect_station_deep(cat, s, months, max_workers=4)
            stage2_results.append(metrics)

    stage2_df = pd.DataFrame(stage2_results)

    # Merge metadata if available
    if not metadata_df.empty:
        meta_sub = metadata_df[["station_code", "station_name", "basin_name", "province_name", "amphoe_name", "latitude", "longitude"]].drop_duplicates("station_code")
        stage2_df = stage2_df.merge(meta_sub, on="station_code", how="left")

    # Generate Golden stations CSV
    golden_df = stage2_df[stage2_df["is_golden"]].copy() if not stage2_df.empty else pd.DataFrame()
    golden_csv_path = output_dir / "golden_stations_zero_null_2021_2025.csv"
    golden_df.to_csv(golden_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved %d Golden stations to %s", len(golden_df), golden_csv_path)

    # Generate Silver stations CSV
    silver_df = stage2_df[stage2_df["tier"] == "Tier 2: Silver"].copy() if not stage2_df.empty else pd.DataFrame()
    silver_csv_path = output_dir / "silver_stations_completeness_report.csv"
    silver_df.to_csv(silver_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved %d Silver stations to %s", len(silver_df), silver_csv_path)

    # Multi-Catalog Joint Intersection
    # A station is Joint Triple-Golden if it is Golden in all 3 catalogs
    joint_rows = []
    if not stage2_df.empty:
        stn_groups = stage2_df.groupby("station_code")
        for s, grp in stn_groups:
            cats = set(grp["catalog"])
            if set(catalogs).issubset(cats):
                all_golden = grp["is_golden"].all()
                all_silver_or_better = (grp["tier"].isin(["Tier 1: Golden", "Tier 2: Silver"])).all()
                row = {
                    "station_code": s,
                    "all_three_golden": all_golden,
                    "all_three_silver_or_better": all_silver_or_better,
                    "min_completeness_pct": grp["completeness_pct"].min(),
                    "avg_completeness_pct": round(grp["completeness_pct"].mean(), 4),
                }
                for c in catalogs:
                    sub_c = grp[grp["catalog"] == c]
                    if not sub_c.empty:
                        row[f"{c}_tier"] = sub_c["tier"].values[0]
                        row[f"{c}_completeness_pct"] = sub_c["completeness_pct"].values[0]
                joint_rows.append(row)

    joint_df = pd.DataFrame(joint_rows)
    if not metadata_df.empty and not joint_df.empty:
        joint_df = joint_df.merge(meta_sub, on="station_code", how="left")
    joint_csv_path = output_dir / "joint_triple_golden_stations.csv"
    joint_df.to_csv(joint_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved %d Joint intersection records to %s", len(joint_df), joint_csv_path)

    # Generate spatial distribution map
    map_png_path = output_dir / "spatial_distribution_map.png"
    generate_spatial_map(stage2_df, metadata_df, map_png_path)

    # Generate markdown report
    report_md_path = output_dir / "audit_summary_report.md"
    write_summary_markdown(meta_summary, stage1_stats, stage2_df, joint_df, report_md_path)

    logger.info("HII Audit completed successfully.")
    return {
        "stage1_stations_60m": stage1_stats,
        "stage2_inspected": len(stage2_df),
        "golden_count": len(golden_df),
        "silver_count": len(silver_df),
        "joint_count": len(joint_df),
    }


def main():
    parser = argparse.ArgumentParser(description="HII Ground Data Acquisition & Completeness Audit")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample", help="Mode: sample (Dev) or full (Prod)")
    parser.add_argument("--limit-stations", type=int, default=5, help="Number of stations to inspect in sample mode")
    parser.add_argument("--months", nargs="+", help="Specific months (e.g. 202101 202102) or default all 60")
    parser.add_argument("--all-catalogs", action="store_true", help="Audit all 3 catalogs")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_AUDIT_DIR), help="Output directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrency workers")
    args = parser.parse_args()

    months = args.months if args.months else ALL_60_MONTHS
    catalogs = ["hourly_rain", "pressure", "humidity"]
    output_dir = Path(args.output_dir)

    run_audit(
        mode=args.mode,
        limit_stations=args.limit_stations,
        months=months,
        catalogs=catalogs,
        output_dir=output_dir,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
