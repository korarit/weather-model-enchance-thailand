"""
HII Data Parser & Validation Engine
Handles:
- Legacy Schema (2021-2024): date,time,<metric>
- Modern Schema (2025): station_code,measure_datetime,<metric>,quality_flag
- Missing, Sentinel, Quality Flag, and Physical Bound Validation
"""

import io
import re
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Meteorological Physical Bounds for Thailand
PHYSICAL_BOUNDS = {
    "hourly_rain": (0.0, 300.0),       # mm/h
    "pressure": (850.0, 1050.0),       # hPa
    "humidity": (0.0, 100.0),          # %
}

SENTINELS = {-999, -999.0, -9999, -9999.0}
NULL_STRINGS = {"", " ", "null", "none", "nan", "na", "n/a", "-"}

# Map raw column names to canonical variable types
COLUMN_VARIABLE_MAP = {
    "rain": "hourly_rain",
    "rainfall_1h": "hourly_rain",
    "rainfall": "hourly_rain",
    "press": "pressure",
    "pressure": "pressure",
    "humid": "humidity",
    "humidity": "humidity",
}


def detect_and_normalize_csv(
    content: bytes,
    station_code: str,
    catalog_hint: str
) -> pd.DataFrame:
    """
    Parses raw CSV bytes from HII and standardizes columns to:
    [station_code, observed_at, variable_type, raw_value, quality_flag]
    """
    if not content or len(content.strip()) == 0:
        return pd.DataFrame(columns=["station_code", "observed_at", "variable_type", "raw_value", "quality_flag"])

    # Try decode
    text = None
    for enc in ["utf-8-sig", "utf-8", "tis-620", "cp874", "latin1"]:
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue

    if text is None:
        raise ValueError("Unable to decode CSV text")

    # Read CSV
    df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    if df.empty:
        return pd.DataFrame(columns=["station_code", "observed_at", "variable_type", "raw_value", "quality_flag"])

    df.columns = [c.strip().lower() for c in df.columns]

    # Detect modern 2025 format: station_code, measure_datetime, ...
    if "measure_datetime" in df.columns:
        dt_col = "measure_datetime"
        observed_at = pd.to_datetime(df[dt_col], errors="coerce")
        qflag = df["quality_flag"].str.strip() if "quality_flag" in df.columns else "VALID"
        stn = df["station_code"].str.strip() if "station_code" in df.columns else station_code

        # Find value column
        val_col = None
        for col in df.columns:
            if col in COLUMN_VARIABLE_MAP:
                val_col = col
                break
        if not val_col:
            # Fallback to 3rd column
            val_col = df.columns[2]

        val_series = df[val_col]
        var_type = COLUMN_VARIABLE_MAP.get(val_col, catalog_hint)

    # Legacy 2021-2024 format: date, time, <val>
    elif "date" in df.columns and "time" in df.columns:
        dt_series = df["date"].str.strip() + " " + df["time"].str.strip()
        observed_at = pd.to_datetime(dt_series, errors="coerce")
        stn = station_code
        qflag = "VALID"

        val_col = None
        for col in df.columns:
            if col not in ("date", "time"):
                val_col = col
                break
        if not val_col:
            raise ValueError(f"Value column not found in legacy CSV for {station_code}")

        val_series = df[val_col]
        var_type = COLUMN_VARIABLE_MAP.get(val_col, catalog_hint)

    else:
        raise ValueError(f"Unrecognized CSV format with columns: {list(df.columns)}")

    result = pd.DataFrame({
        "station_code": stn,
        "observed_at": observed_at,
        "variable_type": var_type,
        "raw_value": val_series,
        "raw_quality_flag": qflag,
    })

    # Drop any rows where datetime could not be parsed
    result = result.dropna(subset=["observed_at"]).sort_values("observed_at").drop_duplicates(subset=["observed_at"])
    return result


def audit_station_records(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    variable_type: str,
    station_code: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Audits parsed observations against a complete hourly timestamp grid.
    Returns:
      - Cleaned and annotated DataFrame
      - Detailed audit metrics dictionary
    """
    expected_index = pd.date_range(start=start_dt, end=end_dt, freq="1h", name="observed_at")
    total_expected = len(expected_index)

    if df.empty:
        # 100% missing
        metrics = {
            "station_code": station_code,
            "variable_type": variable_type,
            "total_expected_hours": total_expected,
            "actual_hours_reported": 0,
            "missing_timestamp_hours": total_expected,
            "explicit_null_hours": 0,
            "sentinel_error_hours": 0,
            "flag_error_hours": 0,
            "out_of_bounds_hours": 0,
            "valid_hours": 0,
            "completeness_pct": 0.0,
            "max_consecutive_gap_hours": total_expected,
            "tier": "Excluded",
            "is_golden": False,
        }
        return pd.DataFrame(), metrics

    # Reindex onto exact hourly grid
    df = df.set_index("observed_at")
    df = df[~df.index.duplicated(keep="last")]
    full_df = df.reindex(expected_index)
    full_df["station_code"] = station_code
    full_df["variable_type"] = variable_type

    # Classify each timestamp
    # 1. Missing timestamp check
    is_missing_ts = full_df["raw_value"].isna()

    # 2. String cleaning and numeric conversion
    raw_str = full_df["raw_value"].astype(str).str.strip().str.lower()
    is_explicit_null = is_missing_ts | raw_str.isin(NULL_STRINGS)

    # Convert to numeric
    numeric_vals = pd.to_numeric(full_df["raw_value"], errors="coerce")
    is_numeric_null = (~is_missing_ts) & numeric_vals.isna()
    is_null_total = is_explicit_null | is_numeric_null

    # 3. Sentinel check (-999, -9999)
    is_sentinel = numeric_vals.isin(SENTINELS)

    # 4. Quality flag check (2025: N = Normal, others like E, M, B are bad)
    raw_flag = full_df["raw_quality_flag"].fillna("").astype(str).str.strip().str.upper()
    is_flag_error = raw_flag.isin(["E", "M", "B", "ERR", "ERROR", "NULL"])

    # 5. Physical bounds check
    min_b, max_b = PHYSICAL_BOUNDS.get(variable_type, (-np.inf, np.inf))
    is_out_of_bounds = (~is_null_total) & (~is_sentinel) & (
        (numeric_vals < min_b) | (numeric_vals > max_b)
    )

    # Determine validity
    is_valid = (~is_null_total) & (~is_sentinel) & (~is_flag_error) & (~is_out_of_bounds)

    # Label status
    status = pd.Series("VALID", index=full_df.index)
    status[is_missing_ts] = "MISSING_TIMESTAMP"
    status[is_explicit_null & (~is_missing_ts)] = "EXPLICIT_NULL"
    status[is_sentinel] = "SENTINEL_ERROR"
    status[is_flag_error] = "FLAG_ERROR"
    status[is_out_of_bounds] = "OUT_OF_BOUNDS"

    full_df["value"] = np.where(is_valid, numeric_vals.astype(np.float32), np.nan)
    full_df["quality_flag"] = status
    full_df["is_valid"] = is_valid

    # Consecutive gap calculation (longest continuous run of invalid or missing hours)
    invalid_mask = (~is_valid).astype(int)
    gap_lengths = []
    current_gap = 0
    for v in invalid_mask:
        if v == 1:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)
    max_gap = max(gap_lengths) if gap_lengths else 0

    valid_count = int(is_valid.sum())
    missing_ts_count = int(is_missing_ts.sum())
    explicit_null_count = int((is_explicit_null & (~is_missing_ts)).sum())
    sentinel_count = int(is_sentinel.sum())
    flag_err_count = int(is_flag_error.sum())
    oob_count = int(is_out_of_bounds.sum())

    completeness_pct = round((valid_count / total_expected) * 100.0, 4)

    # Tier assignment as defined in Phase 1 plan:
    # Tier 1 (Golden): 100% complete, 0 null, 0 sentinel, 0 error flag, 0 out-of-bounds
    # Tier 2 (Silver): >= 99.5% complete, max consecutive gap <= 3 hours
    # Tier 3 (Bronze): 95.0% - 99.4%
    # Tier 4 (Excluded): < 95.0%
    if valid_count == total_expected:
        tier = "Tier 1: Golden"
        is_golden = True
    elif completeness_pct >= 99.5 and max_gap <= 3:
        tier = "Tier 2: Silver"
        is_golden = False
    elif completeness_pct >= 95.0:
        tier = "Tier 3: Bronze"
        is_golden = False
    else:
        tier = "Tier 4: Excluded"
        is_golden = False

    metrics = {
        "station_code": station_code,
        "variable_type": variable_type,
        "total_expected_hours": total_expected,
        "actual_hours_reported": len(df),
        "missing_timestamp_hours": missing_ts_count,
        "explicit_null_hours": explicit_null_count,
        "sentinel_error_hours": sentinel_count,
        "flag_error_hours": flag_err_count,
        "out_of_bounds_hours": oob_count,
        "valid_hours": valid_count,
        "completeness_pct": completeness_pct,
        "max_consecutive_gap_hours": max_gap,
        "tier": tier,
        "is_golden": is_golden,
    }

    full_df = full_df.reset_index().rename(columns={"index": "observed_at"})
    full_df["is_golden"] = is_golden
    return full_df, metrics
