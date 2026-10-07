"""
Thailand 2 km x 2 km Master Grid Generator & Topographic Modeler
Generates:
- Projected UTM Zone 47N (EPSG:32647) 2,000 m x 2,000 m regular grid
- Geographic Lat/Lon (EPSG:4326) centroids
- Topographic attributes (Elevation, Roughness, Slope, Aspect, Distance to Coast, Coriolis)
- Outputs: data/geo/thailand_2km_master_grid.parquet and .csv
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import pyproj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.config.paths import DEFAULT_GEO_DIR, DEFAULT_DEM_DIR
from src.features.copernicus_dem import extract_copernicus_topography

# Geographic Bounds of Thailand
LAT_MIN, LAT_MAX = 5.5, 20.5
LON_MIN, LON_MAX = 97.3, 105.7

# Earth angular velocity (rad/s)
OMEGA_EARTH = 7.2921159e-5


def get_transformers() -> Tuple[pyproj.Transformer, pyproj.Transformer]:
    """Returns bidirectional coordinate transformers between WGS84 and UTM 47N."""
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    to_wgs = pyproj.Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs


def estimate_synthetic_topography(lat: np.ndarray, lon: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes elevation and topographic parameters using meteorologically realistic
    terrain patterns of Thailand:
    - Northern mountain ranges (Thanon Thong Chai, Daen Lao, Luang Prabang: 500-2500m)
    - Western Tenasserim range (300-1800m)
    - Central Chao Phraya alluvial plain (2-30m)
    - Khorat Plateau (Isan rim: Dong Phaya Yen / Sankamphaeng: 150-800m)
    - Southern Peninsula (Banthad / Sankalakhiri: 50-1400m)
    """
    n = len(lat)
    elev = np.zeros(n, dtype=np.float32)

    # 1. Northern Highlands (lat > 17.5, lon 98 - 101.5)
    mask_north = lat > 17.5
    elev[mask_north] += (
        300.0 + (lat[mask_north] - 17.5) * 200.0 +
        350.0 * np.sin((lon[mask_north] - 98.0) * np.pi / 2.0) ** 2
    )

    # 2. Western Mountain Ridge (Tenasserim: lon < 99.5, lat between 11.5 and 17.5)
    mask_west = (lon < 99.5) & (lat >= 11.5) & (lat <= 17.5)
    elev[mask_west] += 250.0 + (99.5 - lon[mask_west]) * 400.0

    # 3. Khorat Plateau Escarpment (Dong Phaya Yen: lon 101.0 - 102.5, lat 14.0 - 16.5)
    mask_plateau = (lon >= 101.0) & (lat >= 14.0) & (lat <= 18.0)
    elev[mask_plateau] += 160.0 + 100.0 * np.sin((lon[mask_plateau] - 101.0) * np.pi / 3.0)

    # 4. Central Chao Phraya Plains (lat 13.5 - 16.5, lon 99.5 - 101.0)
    mask_central = (lat >= 13.5) & (lat <= 16.5) & (lon >= 99.8) & (lon <= 100.8)
    elev[mask_central] = np.clip(elev[mask_central], 4.0, 45.0)

    # Base elevation noise & clipping
    elev = np.clip(elev + np.random.normal(0, 10.0, size=n), 1.0, 2565.0).astype(np.float32)

    # Topographic roughness (std) and slope
    elevation_std = np.clip(elev * 0.12 + np.random.uniform(2.0, 15.0, size=n), 1.0, 250.0).astype(np.float32)
    slope_deg = np.clip(np.arctan(elevation_std / 2000.0) * (180.0 / np.pi) * 3.5, 0.1, 45.0).astype(np.float32)

    # Aspect orientation (windward south-west monsoon vs leeward north-east)
    aspect_rad = np.random.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    # Distance to Coast (km): approx distance from Gulf of Thailand (lat 12.5, lon 100.5) or Andaman
    dist_gulf = np.sqrt(((lat - 12.5) * 111.0) ** 2 + ((lon - 100.5) * 105.0) ** 2)
    dist_andaman = np.maximum(0.0, (lon - 98.0) * 105.0)
    dist_coast_km = np.clip(np.minimum(dist_gulf, dist_andaman), 0.5, 750.0).astype(np.float32)

    # Coriolis parameter: f = 2 * Omega * sin(lat_rad)
    lat_rad = np.radians(lat)
    coriolis = (2.0 * OMEGA_EARTH * np.sin(lat_rad)).astype(np.float32)

    return {
        "elevation_m": np.round(elev, 1),
        "elevation_std": np.round(elevation_std, 1),
        "slope_deg": np.round(slope_deg, 2),
        "aspect_sin": np.round(aspect_sin, 4),
        "aspect_cos": np.round(aspect_cos, 4),
        "dist_coast_km": np.round(dist_coast_km, 1),
        "coriolis_param": coriolis,
    }


def generate_master_grid(
    step_meters: float = 2000.0,
    sample_subsample: int = 1,
    output_dir: Path = DEFAULT_GEO_DIR,
    dem_source: str = "copernicus",
    dem_dir: Path = DEFAULT_DEM_DIR,
    max_dem_tiles: Optional[int] = None,
    allow_streaming: bool = False,
) -> pd.DataFrame:
    """
    Generates regular Thailand grid in UTM 47N and transforms to WGS84 Lat/Lon.
    Computes topographic features from Copernicus DEM 30m or synthetic baseline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Thailand Master Grid [spacing=%.1f m, subsample=%d, dem_source=%s]...",
                step_meters, sample_subsample, dem_source)

    to_utm, to_wgs = get_transformers()

    # Project bounding box corners to UTM
    x_min, y_min = to_utm.transform(LON_MIN, LAT_MIN)
    x_max, y_max = to_utm.transform(LON_MAX, LAT_MAX)

    logger.info("UTM 47N Extents: X[%.0f - %.0f], Y[%.0f - %.0f]", x_min, x_max, y_min, y_max)

    # Generate regular 2,000m coordinates
    step = step_meters * sample_subsample
    x_coords = np.arange(x_min, x_max, step)
    y_coords = np.arange(y_min, y_max, step)

    xv, yv = np.meshgrid(x_coords, y_coords)
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()

    logger.info("Raw rectangular grid points: %d", len(xv_flat))

    # Transform to Lat/Lon
    lons, lats = to_wgs.transform(xv_flat, yv_flat)

    # Approximate Thailand land polygon bounding mask
    # Filter points roughly within Thailand's borders
    valid_mask = (
        (lats >= 5.6) & (lats <= 20.4) &
        (lons >= 97.4) & (lons <= 105.6)
    )

    # Narrow down peninsula vs mainland
    peninsula_mask = (lats < 12.0) & (lons >= 98.2) & (lons <= 102.0)
    mainland_mask = (lats >= 12.0)
    thailand_mask = valid_mask & (peninsula_mask | mainland_mask)

    x_selected = xv_flat[thailand_mask]
    y_selected = yv_flat[thailand_mask]
    lat_selected = lats[thailand_mask]
    lon_selected = lons[thailand_mask]

    n_cells = len(x_selected)
    logger.info("Filtered Thailand land grid cells: %d cells", n_cells)

    # Generate Unique Cell IDs
    cell_ids = [f"GRID_{i+1:06d}" for i in range(n_cells)]

    # Compute Topography (Copernicus DEM 30m or Synthetic Baseline)
    if dem_source == "copernicus":
        logger.info("Extracting topography using Copernicus DEM 30m (dem_dir=%s)...", dem_dir)
        topo_attrs = extract_copernicus_topography(
            lat_selected,
            lon_selected,
            dem_dir=dem_dir,
            allow_streaming=allow_streaming,
            max_tiles=max_dem_tiles,
        )
    else:
        logger.info("Estimating topography using Synthetic mathematical terrain model...")
        topo_attrs = estimate_synthetic_topography(lat_selected, lon_selected)

    df_grid = pd.DataFrame({
        "grid_id": cell_ids,
        "centroid_lat": np.round(lat_selected, 5),
        "centroid_lon": np.round(lon_selected, 5),
        "utm_x": np.round(x_selected, 1),
        "utm_y": np.round(y_selected, 1),
        "elevation_m": topo_attrs["elevation_m"],
        "elevation_std": topo_attrs["elevation_std"],
        "slope_deg": topo_attrs["slope_deg"],
        "aspect_sin": topo_attrs["aspect_sin"],
        "aspect_cos": topo_attrs["aspect_cos"],
        "dist_coast_km": topo_attrs["dist_coast_km"],
        "coriolis_param": topo_attrs["coriolis_param"],
    })

    # Save Parquet and CSV
    parquet_out = output_dir / "thailand_2km_master_grid.parquet"
    csv_out = output_dir / "thailand_2km_master_grid.csv"

    df_grid.to_parquet(parquet_out, index=False, compression="snappy")
    df_grid.to_csv(csv_out, index=False)
    logger.info("Master Grid saved: %s and %s (%d cells)", parquet_out.name, csv_out.name, len(df_grid))

    return df_grid


def main():
    parser = argparse.ArgumentParser(description="Thailand 2 km Master Grid Generator")
    parser.add_argument("--step-meters", type=float, default=2000.0, help="Grid step in meters (default 2000m)")
    parser.add_argument("--sample-subsample", type=int, default=1, help="Subsample step factor for dev mode (e.g. 5 for rapid preview, 1 for full 2km)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_GEO_DIR), help="Output directory")
    parser.add_argument("--dem-source", choices=["copernicus", "synthetic"], default="copernicus", help="Topography DEM source (copernicus or synthetic)")
    parser.add_argument("--dem-dir", type=str, default=str(DEFAULT_DEM_DIR), help="Directory containing Copernicus DEM GeoTIFF files")
    parser.add_argument("--max-dem-tiles", type=int, default=None, help="Max Copernicus tiles to process (useful for rapid testing)")
    parser.add_argument("--allow-streaming", action="store_true", help="Allow direct HTTP streaming from AWS S3 if tile is not cached locally")
    args = parser.parse_args()

    generate_master_grid(
        step_meters=args.step_meters,
        sample_subsample=args.sample_subsample,
        output_dir=Path(args.output_dir),
        dem_source=args.dem_source,
        dem_dir=Path(args.dem_dir),
        max_dem_tiles=args.max_dem_tiles,
        allow_streaming=args.allow_streaming,
    )


if __name__ == "__main__":
    main()
