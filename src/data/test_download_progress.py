"""
Tests for Himawari & Hybrid NWP Download Progress, Per-Model Breakdown, and Skip Tracking.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data.himawari_extractor import run_himawari_acquisition, get_satellite_model_name
from src.data.hybrid_downloader import run_hybrid_batch


def test_himawari_satellite_model_detection():
    # 2021 is Himawari-8 (H08)
    dt_h8 = pd.Timestamp("2021-06-01 12:00:00")
    assert "Himawari-8" in get_satellite_model_name(dt_h8)
    assert "H08" in get_satellite_model_name(dt_h8)

    # 2024 is Himawari-9 (H09)
    dt_h9 = pd.Timestamp("2024-06-01 12:00:00")
    assert "Himawari-9" in get_satellite_model_name(dt_h9)
    assert "H09" in get_satellite_model_name(dt_h9)


def test_himawari_acquisition_progress_and_skip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # First run: should newly extract/generate all 2 snapshots
        res1 = run_himawari_acquisition(
            start_dt=pd.Timestamp("2021-01-01 00:00:00"),
            sample_hours=2,
            step_hours=1,
            source="synthetic",
            out_dir=out_dir,
            overwrite=False
        )
        assert len(res1) == 2
        for f in res1:
            assert f.exists()

        # Second run: should recognize existing files as cached and skip them cleanly
        res2 = run_himawari_acquisition(
            start_dt=pd.Timestamp("2021-01-01 00:00:00"),
            sample_hours=2,
            step_hours=1,
            source="synthetic",
            out_dir=out_dir,
            overwrite=False
        )
        assert len(res2) == 2
        assert res1 == res2


def test_himawari_model_filter():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        # Run with filter for Himawari-9 over a 2021 date range (which is Himawari-8)
        # Should result in 0 snapshots processed
        res = run_himawari_acquisition(
            start_dt=pd.Timestamp("2021-01-01 00:00:00"),
            sample_hours=2,
            models="himawari9",
            source="synthetic",
            out_dir=out_dir
        )
        assert len(res) == 0
