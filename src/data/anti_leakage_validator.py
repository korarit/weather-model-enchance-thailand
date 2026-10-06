"""
Temporal Anti-Leakage Protocol & Look-ahead Prevention Validator
Enforces the Meteorological Look-ahead Prevention Invariant:
At any forecast_run_time T_run, ML bias correction models have access ONLY to
observations (ground & satellite) recorded at or before T_run:
  observation_time <= forecast_run_time

Any observation timestamp where observation_time > forecast_run_time is strictly forbidden
and triggers an immediate TemporalDataLeakageError.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Iterable, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalDataLeakageError(Exception):
    """Raised when an observation timestamp occurs after the forecast run time."""
    pass


def assert_no_temporal_leakage(
    run_time: Union[str, pd.Timestamp],
    observation_timestamps: Union[pd.Series, Iterable[pd.Timestamp]],
    context: str = "general"
) -> bool:
    """
    Validates that NO observation timestamp exceeds the forecast run time.
    Raises TemporalDataLeakageError if any violation occurs.
    """
    run_ts = pd.to_datetime(run_time)
    if isinstance(observation_timestamps, pd.Series):
        obs_series = pd.to_datetime(observation_timestamps)
    else:
        obs_series = pd.to_datetime(pd.Series(list(observation_timestamps)))

    violating_mask = obs_series > run_ts
    violating_count = int(violating_mask.sum())

    if violating_count > 0:
        violating_samples = obs_series[violating_mask].head(5).tolist()
        msg = (
            f"[TEMPORAL LEAKAGE DETECTED in '{context}'] "
            f"Forecast run_time is {run_ts}. Found {violating_count} observations "
            f"recorded AFTER run_time! Violating timestamps: {violating_samples}"
        )
        logger.critical(msg)
        raise TemporalDataLeakageError(msg)

    logger.debug("Temporal anti-leakage verified for run_time %s (%d observations checked)", run_ts, len(obs_series))
    return True


def enforce_anti_leakage_filter(
    run_time: Union[str, pd.Timestamp],
    df: pd.DataFrame,
    time_column: str = "observed_at",
    max_lookback_hours: int = 24
) -> pd.DataFrame:
    """
    Strictly filters a dataset to keep only observations within [run_time - lookback, run_time].
    Guarantees zero future data leakage into downstream feature extractors.
    """
    run_ts = pd.to_datetime(run_time)
    min_ts = run_ts - pd.Timedelta(hours=max_lookback_hours)

    clean_df = df.copy()
    clean_df[time_column] = pd.to_datetime(clean_df[time_column])

    # Filter
    valid_mask = (clean_df[time_column] <= run_ts) & (clean_df[time_column] >= min_ts)
    filtered_df = clean_df[valid_mask].copy()

    # Re-verify invariant
    assert_no_temporal_leakage(run_ts, filtered_df[time_column], context="filtered_observation_window")
    return filtered_df


def run_unit_tests():
    """Unit test suite verifying look-ahead prevention invariants."""
    print("Running Temporal Anti-Leakage Invariant Test Suite...")
    run_time = pd.Timestamp("2025-06-01 00:00:00")

    # 1. Test Valid Historical Timestamps
    valid_times = pd.Series([
        pd.Timestamp("2025-05-31 23:30:00"),
        pd.Timestamp("2025-05-31 23:40:00"),
        pd.Timestamp("2025-05-31 23:50:00"),
        pd.Timestamp("2025-06-01 00:00:00"),  # Exact run time is allowed
    ])
    assert assert_no_temporal_leakage(run_time, valid_times, context="valid_test") is True
    print("  [PASS] Valid historical observations <= run_time correctly accepted.")

    # 2. Test Single Future Timestamp (10 minutes after run_time)
    invalid_times_10m = pd.Series([
        pd.Timestamp("2025-06-01 00:00:00"),
        pd.Timestamp("2025-06-01 00:10:00"),  # Leakage!
    ])
    try:
        assert_no_temporal_leakage(run_time, invalid_times_10m, context="invalid_10m")
        assert False, "Failed to catch 10m future leakage!"
    except TemporalDataLeakageError:
        print("  [PASS] Caught future leakage (+10m) with TemporalDataLeakageError.")

    # 3. Test Forecast Valid Time Leakage (+6 hours)
    invalid_times_valid_time = pd.Series([
        pd.Timestamp("2025-06-01 06:00:00"),  # Valid time of forecast, not run time!
    ])
    try:
        assert_no_temporal_leakage(run_time, invalid_times_valid_time, context="invalid_valid_time")
        assert False, "Failed to catch forecast valid time observation leakage!"
    except TemporalDataLeakageError:
        print("  [PASS] Caught forecast valid time observation (+6h) with TemporalDataLeakageError.")

    # 4. Test enforce_anti_leakage_filter
    test_df = pd.DataFrame({
        "observed_at": [
            pd.Timestamp("2025-05-30 00:00:00"),  # Too old (> 24h)
            pd.Timestamp("2025-05-31 12:00:00"),  # Valid
            pd.Timestamp("2025-06-01 00:00:00"),  # Valid (exact)
            pd.Timestamp("2025-06-01 01:00:00"),  # Future (leakage)
        ],
        "value": [1, 2, 3, 4]
    })
    filtered = enforce_anti_leakage_filter(run_time, test_df, time_column="observed_at", max_lookback_hours=24)
    assert len(filtered) == 2, f"Expected 2 rows, got {len(filtered)}"
    assert (filtered["observed_at"] <= run_time).all()
    print("  [PASS] enforce_anti_leakage_filter successfully pruned future and out-of-window records.")

    print("All Anti-Leakage Unit Tests Passed Successfully! (4/4)")


if __name__ == "__main__":
    run_unit_tests()
