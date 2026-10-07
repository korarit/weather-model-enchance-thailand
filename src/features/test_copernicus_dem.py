"""
Unit Tests for Copernicus DEM 30m Downloader & Topography Extractor
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.copernicus_dem import (
    format_tile_name,
    format_tile_url,
    get_thailand_tile_list,
    compute_terrain_gradients,
    compute_geographic_proxies,
    extract_copernicus_topography,
)
from src.config.paths import DEFAULT_DEM_DIR


def test_tile_name_and_url_formatting():
    """Verifies standard AWS Open Data Copernicus tile name and URL."""
    name = format_tile_name(18, 98)
    assert name == "Copernicus_DSM_COG_10_N18_00_E098_00_DEM"
    
    url = format_tile_url(18, 98)
    assert url == "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N18_00_E098_00_DEM/Copernicus_DSM_COG_10_N18_00_E098_00_DEM.tif"


def test_thailand_tile_bounds():
    """Verifies that tile coordinates cover Thailand."""
    tiles = get_thailand_tile_list(lat_min=5, lat_max=20, lon_min=97, lon_max=105)
    assert len(tiles) == 16 * 9  # 144 candidate 1x1 deg tiles
    assert (13, 100) in tiles  # Bangkok
    assert (18, 98) in tiles   # Chiang Mai


def test_terrain_gradients_slope_and_aspect():
    """Tests that slope and aspect correctly compute on synthetic ramp."""
    # Synthetic terrain: constant slope in X direction: dz/dx = 0.5, dz/dy = 0
    # res = 30m, so delta_z per 30m is 15m
    grid = np.zeros((10, 10), dtype=np.float32)
    for j in range(10):
        grid[:, j] = j * 15.0

    slope, sin_asp, cos_asp = compute_terrain_gradients(grid, res_meters=30.0)
    # Expected slope: arctan(15/30) = arctan(0.5) ~ 26.565 deg
    assert np.isclose(slope[5, 5], 26.565, atol=1.0)
    # sin^2 + cos^2 should be ~ 1.0
    assert np.isclose(sin_asp[5, 5]**2 + cos_asp[5, 5]**2, 1.0, atol=1e-4)


def test_geographic_proxies():
    """Tests distance to coast and Coriolis parameter calculation."""
    lats = np.array([13.75, 18.78], dtype=np.float32)
    lons = np.array([100.5, 98.98], dtype=np.float32)
    dist, coriolis = compute_geographic_proxies(lats, lons)

    assert len(dist) == 2
    assert len(coriolis) == 2
    assert np.all(dist > 0.0)
    # Higher latitude should have higher Coriolis parameter
    assert coriolis[1] > coriolis[0]


def test_extraction_end_to_end():
    """Tests end-to-end extraction on sample points."""
    lats = np.array([18.788, 13.750], dtype=np.float32)
    lons = np.array([98.985, 100.500], dtype=np.float32)
    
    topo = extract_copernicus_topography(lats, lons, dem_dir=DEFAULT_DEM_DIR)
    
    assert "elevation_m" in topo
    assert "elevation_std" in topo
    assert "slope_deg" in topo
    assert "aspect_sin" in topo
    assert "aspect_cos" in topo
    assert "dist_coast_km" in topo
    assert "coriolis_param" in topo
    assert len(topo["elevation_m"]) == 2
    # Elevations should be realistic positive numbers
    assert np.all(topo["elevation_m"] > 0)
