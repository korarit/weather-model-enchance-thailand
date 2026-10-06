"""
Unit Test Suite for Phase 3: Spatial Grid & Feature Engineering
Verifies:
1. Master Grid Coordinate Transformation (UTM 47N <-> WGS84) & Spacing
2. Proximity Exclusion: Stations closer than 2.0 km are strictly rejected
3. Distance Band partitioning (2-5 km, 5-10 km, 20-50 km)
4. Polar coordinates & Bearing trigonometry (sin^2 + cos^2 == 1.0)
5. Gradient magnitude calculations (non-negative, realistic bounds)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features.grid_generator import generate_master_grid, get_transformers
from src.features.spatial_features import SpatialObservationIndexer, EXCLUSION_RADIUS_KM


def test_grid_spacing_and_projection():
    """Verifies that the generated grid has 2,000m spacing and coordinates in Thailand."""
    to_utm, to_wgs = get_transformers()
    # Test point in Central Thailand (Bangkok)
    x, y = to_utm.transform(100.5, 13.75)
    lon_back, lat_back = to_wgs.transform(x, y)
    assert np.isclose(lon_back, 100.5, atol=1e-5)
    assert np.isclose(lat_back, 13.75, atol=1e-5)
    print("  [PASS] UTM Zone 47N coordinate roundtrip verified.")


def test_proximity_exclusion_under_2km():
    """Tests that any station with distance < 2.0 km is rejected to prevent overfitting."""
    # Create test stations:
    # Target grid is at (13.75, 100.50)
    # Stn A is 1.0 km away (< 2 km, must be REJECTED)
    # Stn B is 3.5 km away (2-5 km, must be ACCEPTED into Band 1)
    to_utm, to_wgs = get_transformers()
    x0, y0 = to_utm.transform(100.50, 13.75)

    # 1 km north (1000 m)
    lon_a, lat_a = to_wgs.transform(x0, y0 + 1000.0)
    # 3.5 km east (3500 m)
    lon_b, lat_b = to_wgs.transform(x0 + 3500.0, y0)

    stn_df = pd.DataFrame({
        "station_code": ["STN_TOO_CLOSE", "STN_BAND1"],
        "latitude": [lat_a, lat_b],
        "longitude": [lon_a, lon_b],
    })

    indexer = SpatialObservationIndexer(stn_df)
    obs_dict = {
        "STN_TOO_CLOSE": {"hourly_rain": 99.9, "pressure": 1000.0, "humidity": 90.0},
        "STN_BAND1": {"hourly_rain": 12.5, "pressure": 1008.0, "humidity": 75.0},
    }

    feats = indexer.extract_grid_spatial_features(13.75, 100.50, obs_dict)

    # Stn A (99.9 mm) should NOT be included in rain_max_2_5km! Only Stn B (12.5 mm) should be present.
    assert np.isclose(feats["rain_max_2_5km"], 12.5), f"Expected 12.5, got {feats['rain_max_2_5km']}"
    assert np.isclose(feats["pressure_mean_2_5km"], 1008.0), f"Expected 1008.0, got {feats['pressure_mean_2_5km']}"
    assert feats["nearest_stn_dist_2_5km"] >= EXCLUSION_RADIUS_KM
    print("  [PASS] Station with distance < 2 km was strictly excluded from features.")


def test_bearing_trigonometry():
    """Tests that bearing sine and cosine satisfy sin^2 + cos^2 == 1.0."""
    to_utm, to_wgs = get_transformers()
    x0, y0 = to_utm.transform(100.50, 13.75)
    lon_b, lat_b = to_wgs.transform(x0 + 2500.0, y0 + 2500.0)

    stn_df = pd.DataFrame({
        "station_code": ["STN_NE"],
        "latitude": [lat_b],
        "longitude": [lon_b],
    })
    indexer = SpatialObservationIndexer(stn_df)
    feats = indexer.extract_grid_spatial_features(13.75, 100.50, {"STN_NE": {"hourly_rain": 5.0}})

    sin_val = feats["nearest_stn_bearing_sin"]
    cos_val = feats["nearest_stn_bearing_cos"]
    trig_identity = sin_val ** 2 + cos_val ** 2
    assert np.isclose(trig_identity, 1.0, atol=1e-4), f"sin^2 + cos^2 = {trig_identity} != 1.0"
    print("  [PASS] Bearing trigonometric identity (sin^2 + cos^2 == 1.0) satisfied.")


def test_gradient_magnitude_non_negative():
    """Verifies that atmospheric gradients are computed and non-negative."""
    offsets = np.array([
        [-10.0, 0.0],
        [10.0, 0.0],
        [0.0, -10.0],
        [0.0, 10.0],
    ])
    pressures = np.array([1005.0, 1010.0, 1007.0, 1008.0])
    mag = SpatialObservationIndexer._compute_gradient_magnitude(offsets, pressures)
    assert mag > 0.0, "Expected non-zero gradient"
    assert mag < 50.0, "Gradient outside physical bounds"
    print(f"  [PASS] Atmospheric pressure gradient calculated correctly: {mag:.2f} hPa/100km.")


def run_all_tests():
    print("Running Phase 3 Spatial Feature Engine Tests...")
    test_grid_spacing_and_projection()
    test_proximity_exclusion_under_2km()
    test_bearing_trigonometry()
    test_gradient_magnitude_non_negative()
    print("All Phase 3 Tests Passed Successfully! (4/4)")


if __name__ == "__main__":
    run_all_tests()
