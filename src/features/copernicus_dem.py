"""
Copernicus DEM GLO-30 (30m Resolution) Topography Acquisition & Feature Extraction
Provides:
- Direct tile acquisition from AWS Open Data (s3://copernicus-dem-30m / HTTPS)
- Batch downloading for Thailand coverage (109 tiles, 1x1 degree COG)
- Precise 30m terrain feature extraction:
    - elevation_m (mean elevation across 2km grid cell)
    - elevation_std (true topographic roughness / standard deviation of 30m DEM pixels)
    - slope_deg (terrain slope in degrees derived via spatial gradient)
    - aspect_sin, aspect_cos (slope orientation / aspect decomposition)
    - dist_coast_km (distance to nearest coastline)
    - coriolis_param (Coriolis parameter 2*Omega*sin(lat))
- Flexible execution modes: Local Cached Files, Remote COG Streaming, or Dry-run/Test mode.
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:
    import rasterio
    import rasterio.windows
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import DEFAULT_GEO_DIR, DEFAULT_DEM_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# AWS Copernicus DEM 30m Open Data Base URL
COPERNICUS_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

# Earth Angular Velocity for Coriolis parameter
OMEGA_EARTH = 7.2921159e-5


def format_tile_name(lat: int, lon: int) -> str:
    """Formats Copernicus DEM 1x1 degree tile identifier."""
    lat_prefix = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_prefix = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{lat_prefix}_00_{lon_prefix}_00_DEM"


def format_tile_url(lat: int, lon: int) -> str:
    """Formats public HTTPS URL for Copernicus DEM COG GeoTIFF on AWS S3."""
    name = format_tile_name(lat, lon)
    return f"{COPERNICUS_BASE_URL}/{name}/{name}.tif"


def get_thailand_tile_list(
    lat_min: int = 5,
    lat_max: int = 20,
    lon_min: int = 97,
    lon_max: int = 105
) -> List[Tuple[int, int]]:
    """Returns grid of 1x1 degree tile coordinates covering Thailand."""
    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append((lat, lon))
    return tiles


def download_copernicus_tile(
    lat: int,
    lon: int,
    dem_dir: Path = DEFAULT_DEM_DIR,
    timeout: int = 45,
    chunk_size: int = 1024 * 1024
) -> Optional[Path]:
    """
    Downloads a single 1x1 degree Copernicus DEM 30m GeoTIFF tile.
    Returns:
        Path to downloaded file if successful or already exists.
        None if tile is pure ocean (HTTP 404) or failed.
    """
    dem_dir.mkdir(parents=True, exist_ok=True)
    tile_name = format_tile_name(lat, lon)
    target_file = dem_dir / f"{tile_name}.tif"
    ocean_marker = dem_dir / f"{tile_name}.ocean"

    # Already downloaded or marked as ocean
    if target_file.exists() and target_file.stat().st_size > 1024:
        return target_file
    if ocean_marker.exists():
        return None

    url = format_tile_url(lat, lon)
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        if resp.status_code == 404:
            # Ocean tile without land elevation data
            ocean_marker.touch()
            return None
        elif resp.status_code != 200:
            logger.warning("Failed to download %s: HTTP %d", tile_name, resp.status_code)
            return None

        temp_file = target_file.with_suffix(".tmp")
        with open(temp_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        temp_file.replace(target_file)
        logger.info("Downloaded Copernicus DEM tile: %s (%.1f MB)", target_file.name, target_file.stat().st_size / 1e6)
        return target_file
    except Exception as e:
        logger.warning("Error downloading tile %s from %s: %s", tile_name, url, e)
        return None


def download_all_thailand_tiles(
    dem_dir: Path = DEFAULT_DEM_DIR,
    max_workers: int = 4,
    limit_tiles: Optional[int] = None
) -> Dict[str, int]:
    """
    Batch downloads Copernicus DEM 30m tiles covering Thailand.
    Ideal for execution in high-bandwidth environments like Google Colab.
    """
    tiles = get_thailand_tile_list()
    if limit_tiles:
        tiles = tiles[:limit_tiles]

    logger.info("Starting batch download of Copernicus DEM tiles (%d total)...", len(tiles))
    downloaded = 0
    ocean_or_missing = 0
    existing = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_copernicus_tile, lat, lon, dem_dir): (lat, lon)
            for lat, lon in tiles
        }
        for future in as_completed(future_map):
            lat, lon = future_map[future]
            res = future.result()
            if res is not None:
                downloaded += 1
            else:
                ocean_or_missing += 1

    logger.info("Copernicus DEM download completed: %d available, %d ocean/missing", downloaded, ocean_or_missing)
    return {"total": len(tiles), "available": downloaded, "ocean_or_missing": ocean_or_missing}


def compute_terrain_gradients(
    elev_grid: np.ndarray,
    res_meters: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes slope and aspect angles from DEM elevation raster using spatial gradients.
    Returns:
        (slope_deg, aspect_sin, aspect_cos)
    """
    gy, gx = np.gradient(elev_grid, res_meters, res_meters)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = (np.arctan2(-gx, gy) % (2.0 * np.pi)).astype(np.float32)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    return slope_deg, aspect_sin, aspect_cos


def compute_geographic_proxies(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes distance to coast and Coriolis parameter.
    """
    # Approximate distance to Gulf of Thailand (12.5N, 100.5E) or Andaman Sea (lon 98.0E)
    dist_gulf = np.sqrt(((lat - 12.5) * 111.0) ** 2 + ((lon - 100.5) * 105.0) ** 2)
    dist_andaman = np.maximum(0.0, (lon - 98.0) * 105.0)
    dist_coast_km = np.clip(np.minimum(dist_gulf, dist_andaman), 0.5, 750.0).astype(np.float32)

    # Coriolis parameter: f = 2 * Omega * sin(lat_rad)
    lat_rad = np.radians(lat)
    coriolis = (2.0 * OMEGA_EARTH * np.sin(lat_rad)).astype(np.float32)

    return dist_coast_km, coriolis


def extract_copernicus_topography(
    lats: np.ndarray,
    lons: np.ndarray,
    dem_dir: Path = DEFAULT_DEM_DIR,
    cell_radius_meters: float = 1000.0,
    allow_streaming: bool = False,
    max_tiles: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Extracts high-resolution topographic attributes from Copernicus DEM 30m for given coordinates.
    For each coordinate:
    - Locates the 1x1 degree Copernicus DEM tile
    - Reads the 2km window around the centroid
    - Calculates mean elevation, elevation std (roughness), slope, aspect_sin, aspect_cos
    - Falls back smoothly to synthetic terrain if tile is not locally available and streaming is disabled.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for Copernicus DEM processing. Install via `pip install rasterio`")

    n = len(lats)
    elev_m = np.zeros(n, dtype=np.float32)
    elev_std = np.zeros(n, dtype=np.float32)
    slope_deg = np.zeros(n, dtype=np.float32)
    aspect_sin = np.zeros(n, dtype=np.float32)
    aspect_cos = np.zeros(n, dtype=np.float32)

    # 1. Compute geographic proxies (coast dist, Coriolis)
    dist_coast_km, coriolis_param = compute_geographic_proxies(lats, lons)

    # 2. Partition coordinate points by 1x1 degree tile
    floor_lats = np.floor(lats).astype(int)
    floor_lons = np.floor(lons).astype(int)
    tile_keys = list(zip(floor_lats, floor_lons))
    unique_tiles = sorted(list(set(tile_keys)))

    processed_tiles = 0
    tile_cache: Dict[Tuple[int, int], Optional[str]] = {}

    for tile_lat, tile_lon in unique_tiles:
        if max_tiles is not None and processed_tiles >= max_tiles:
            break

        tile_name = format_tile_name(tile_lat, tile_lon)
        local_path = dem_dir / f"{tile_name}.tif"
        ocean_marker = dem_dir / f"{tile_name}.ocean"

        tile_source: Optional[str] = None
        if local_path.exists() and local_path.stat().st_size > 1024:
            tile_source = str(local_path)
        elif ocean_marker.exists():
            tile_source = None
        elif allow_streaming:
            tile_source = format_tile_url(tile_lat, tile_lon)

        # Get indices of points in this tile
        idx_mask = np.where((floor_lats == tile_lat) & (floor_lons == tile_lon))[0]
        if len(idx_mask) == 0:
            continue

        if tile_source is None:
            # Tile is not present locally and streaming not enabled (or ocean tile)
            continue

        try:
            with rasterio.open(tile_source) as src:
                processed_tiles += 1
                logger.info("Processing Copernicus DEM tile (%d, %d) with %d grid points...", tile_lat, tile_lon, len(idx_mask))
                
                # Window size: 2km cell has approx 2000m / 30m ~ 66 pixels
                half_pixels = int(round(cell_radius_meters / 30.0))

                for idx in idx_mask:
                    lat_val = lats[idx]
                    lon_val = lons[idx]

                    try:
                        row, col = src.index(lon_val, lat_val)
                    except Exception:
                        continue

                    r_min = max(0, row - half_pixels)
                    r_max = min(src.height, row + half_pixels + 1)
                    c_min = max(0, col - half_pixels)
                    c_max = min(src.width, col + half_pixels + 1)

                    if r_max <= r_min or c_max <= c_min:
                        continue

                    win = rasterio.windows.Window(c_min, r_min, c_max - c_min, r_max - r_min)
                    elev_patch = src.read(1, window=win).astype(np.float32)

                    # Mask nodata (Copernicus DEM nodata value is often -999999 or < -100)
                    valid_mask = (elev_patch > -100.0) & (elev_patch < 9000.0)
                    if not np.any(valid_mask):
                        continue

                    valid_elev = elev_patch[valid_mask]
                    mean_val = float(np.mean(valid_elev))
                    std_val = float(np.std(valid_elev)) if len(valid_elev) > 1 else 0.0

                    # Compute gradient for this patch if large enough
                    if elev_patch.shape[0] >= 3 and elev_patch.shape[1] >= 3:
                        s_grid, sin_grid, cos_grid = compute_terrain_gradients(elev_patch, 30.0)
                        s_val = float(np.mean(s_grid[valid_mask]))
                        sin_val = float(np.mean(sin_grid[valid_mask]))
                        cos_val = float(np.mean(cos_grid[valid_mask]))
                    else:
                        s_val = 0.0
                        sin_val = 0.0
                        cos_val = 0.0

                    elev_m[idx] = mean_val
                    elev_std[idx] = std_val
                    slope_deg[idx] = s_val
                    aspect_sin[idx] = sin_val
                    aspect_cos[idx] = cos_val

        except Exception as e:
            logger.warning("Could not read Copernicus DEM tile (%d, %d): %s", tile_lat, tile_lon, e)
            continue

    # 3. For any coordinates not covered by downloaded DEM tiles, smoothly fill with synthetic model
    missing_mask = (elev_m == 0.0) & (elev_std == 0.0)
    if np.any(missing_mask):
        logger.info("Filling %d grid points not yet downloaded with synthetic topography baseline...", np.sum(missing_mask))
        from src.features.grid_generator import estimate_synthetic_topography
        synth = estimate_synthetic_topography(lats[missing_mask], lons[missing_mask])
        elev_m[missing_mask] = synth["elevation_m"]
        elev_std[missing_mask] = synth["elevation_std"]
        slope_deg[missing_mask] = synth["slope_deg"]
        aspect_sin[missing_mask] = synth["aspect_sin"]
        aspect_cos[missing_mask] = synth["aspect_cos"]

    return {
        "elevation_m": np.round(elev_m, 1),
        "elevation_std": np.round(elev_std, 1),
        "slope_deg": np.round(slope_deg, 2),
        "aspect_sin": np.round(aspect_sin, 4),
        "aspect_cos": np.round(aspect_cos, 4),
        "dist_coast_km": np.round(dist_coast_km, 1),
        "coriolis_param": coriolis_param,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Copernicus DEM 30m Downloader & Topography Processor")
    parser.add_argument("--download-all", action="store_true", help="Download all tiles for Thailand (109 tiles, ~3GB total, for Colab)")
    parser.add_argument("--test-tile", action="store_true", help="Download and test 1 sample tile (Chiang Mai N18 E098) without full download")
    parser.add_argument("--dem-dir", type=str, default=str(DEFAULT_DEM_DIR), help="Directory to store DEM GeoTIFF files")
    parser.add_argument("--max-workers", type=int, default=4, help="Download concurrency threads")
    parser.add_argument("--limit-tiles", type=int, default=None, help="Limit number of tiles to download")
    args = parser.parse_args()

    dem_path = Path(args.dem_dir)
    dem_path.mkdir(parents=True, exist_ok=True)

    if args.test_tile:
        logger.info("Testing download of single Copernicus DEM tile (N18, E098: Chiang Mai)...")
        p = download_copernicus_tile(18, 98, dem_dir=dem_path)
        if p and p.exists():
            logger.info("Successfully fetched test tile: %s (%.1f MB)", p.name, p.stat().st_size / 1e6)
            # Run quick extraction test on 5 sample coordinates in Chiang Mai
            test_lats = np.array([18.788, 18.800, 18.850, 18.900, 18.950], dtype=np.float32)
            test_lons = np.array([98.985, 98.970, 98.920, 98.900, 98.850], dtype=np.float32)
            topo = extract_copernicus_topography(test_lats, test_lons, dem_dir=dem_path)
            for i in range(len(test_lats)):
                print(f"Sample {i+1} ({test_lats[i]:.3f}, {test_lons[i]:.3f}): "
                      f"elev={topo['elevation_m'][i]}m, std={topo['elevation_std'][i]}m, "
                      f"slope={topo['slope_deg'][i]} deg, aspect_sin={topo['aspect_sin'][i]}")
        else:
            logger.error("Failed to fetch test tile.")

    elif args.download_all or args.limit_tiles:
        res = download_all_thailand_tiles(
            dem_dir=dem_path,
            max_workers=args.max_workers,
            limit_tiles=args.limit_tiles
        )
        print(f"Download results: {res}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
