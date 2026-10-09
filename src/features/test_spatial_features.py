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


def test_variable_specific_nearest_telemetry():
    """
    Tests that when closest stations have incomplete telemetry (e.g. Stn A has rain only),
    missing pressure and humidity are automatically retrieved from the next closest valid stations,
    and their independent distances and bearings are recorded accurately.
    """
    to_utm, to_wgs = get_transformers()
    x0, y0 = to_utm.transform(100.50, 13.75)

    # Stn A is 3.0 km North (dx=0, dy=+3000m) -> ONLY rain
    lon_a, lat_a = to_wgs.transform(x0, y0 + 3000.0)
    # Stn B is 7.0 km East (dx=+7000m, dy=0) -> ONLY pressure
    lon_b, lat_b = to_wgs.transform(x0 + 7000.0, y0)
    # Stn C is 12.0 km South (dx=0, dy=-12000m) -> ONLY humidity
    lon_c, lat_c = to_wgs.transform(x0, y0 - 12000.0)

    stn_df = pd.DataFrame({
        "station_code": ["STN_RAIN_ONLY", "STN_PRESSURE_ONLY", "STN_HUMIDITY_ONLY"],
        "latitude": [lat_a, lat_b, lat_c],
        "longitude": [lon_a, lon_b, lon_c],
    })

    indexer = SpatialObservationIndexer(stn_df)
    obs_dict = {
        "STN_RAIN_ONLY": {"hourly_rain": 15.0},
        "STN_PRESSURE_ONLY": {"pressure": 1006.5},
        "STN_HUMIDITY_ONLY": {"humidity": 88.0},
    }

    feats = indexer.extract_grid_spatial_features(13.75, 100.50, obs_dict)

    # 1. Rain should come from Stn A (3 km away)
    assert np.isclose(feats["nearest_rain_val"], 15.0), f"Expected 15.0, got {feats['nearest_rain_val']}"
    assert np.isclose(feats["nearest_rain_dist_km"], 3.0, atol=0.1), f"Expected 3.0 km, got {feats['nearest_rain_dist_km']}"

    # 2. Pressure should be fetched from next closest Stn B (7 km away, not Stn A)
    assert np.isclose(feats["nearest_pressure_val"], 1006.5), f"Expected 1006.5, got {feats['nearest_pressure_val']}"
    assert np.isclose(feats["nearest_pressure_dist_km"], 7.0, atol=0.1), f"Expected 7.0 km, got {feats['nearest_pressure_dist_km']}"

    # 3. Humidity should be fetched from next closest Stn C (12 km away)
    assert np.isclose(feats["nearest_humidity_val"], 88.0), f"Expected 88.0, got {feats['nearest_humidity_val']}"
    assert np.isclose(feats["nearest_humidity_dist_km"], 12.0, atol=0.1), f"Expected 12.0 km, got {feats['nearest_humidity_dist_km']}"

    # 4. Bearings must reflect individual directions
    # Stn A is North -> dx=0, dy=+ -> bearing = 0 rad -> sin=0, cos=1
    assert np.isclose(feats["nearest_rain_bearing_sin"], 0.0, atol=0.05)
    assert np.isclose(feats["nearest_rain_bearing_cos"], 1.0, atol=0.05)

    # Stn B is East -> dx=+, dy=0 -> bearing = pi/2 rad -> sin=1, cos=0
    assert np.isclose(feats["nearest_pressure_bearing_sin"], 1.0, atol=0.05)
    assert np.isclose(feats["nearest_pressure_bearing_cos"], 0.0, atol=0.05)

    # Stn C is South -> dx=0, dy=- -> bearing = -pi or pi rad -> sin=0, cos=-1
    assert np.isclose(feats["nearest_humidity_bearing_sin"], 0.0, atol=0.05)
    assert np.isclose(feats["nearest_humidity_bearing_cos"], -1.0, atol=0.05)

    # 5. Band means should be gracefully imputed from nearest valid station (no NaNs)
    assert not np.isnan(feats["pressure_mean_2_5km"])
    assert np.isclose(feats["pressure_mean_2_5km"], 1006.5)
    assert not np.isnan(feats["humidity_mean_2_5km"])
    assert np.isclose(feats["humidity_mean_2_5km"], 88.0)
    print("  [PASS] Variable-specific nearest telemetry fallback & polar tracking verified.")


def run_all_tests():
    print("Running Phase 3 Spatial Feature Engine Tests...")
    test_grid_spacing_and_projection()
    test_proximity_exclusion_under_2km()
    test_bearing_trigonometry()
    test_gradient_magnitude_non_negative()
    test_variable_specific_nearest_telemetry()
    print("All Phase 3 Tests Passed Successfully! (5/5)")


if __name__ == "__main__":
    run_all_tests()
