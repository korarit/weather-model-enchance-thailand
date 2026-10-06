"""
Spatial Observation Features & Atmospheric Gradient Engine
Implements:
- KDTree Spatial Indexing for Ground Weather Stations
- Strict Proximity Exclusion: Rejects any station with distance < 2.0 km to prevent label leakage
- Distance Bands: Band 1 (2-5 km), Band 2 (5-10 km), Band 3 (10-20 km), Band 4 (20-50 km)
- Polar Geometric Features (Euclidean distance, bearing azimuth theta, sin(theta), cos(theta))
- Atmospheric Spatial Gradients (Pressure Gradient Vector magnitude, Humidity Gradient magnitude)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pyproj

logger = logging.getLogger(__name__)

# Distance band boundaries (km)
EXCLUSION_RADIUS_KM = 2.0
BAND_1_MAX_KM = 5.0
BAND_2_MAX_KM = 10.0
BAND_3_MAX_KM = 20.0
BAND_4_MAX_KM = 50.0


class SpatialObservationIndexer:
    """
    Manages fast KDTree spatial indexing and multi-band feature extraction
    from ground observation telemetry to target 2 km grid cells.
    """

    def __init__(self, stations_metadata_df: pd.DataFrame):
        """
        Initializes spatial indexer using station master metadata (must contain station_code, lat, lon).
        """
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
        
        # Prepare station positions
        stn_df = stations_metadata_df.dropna(subset=["latitude", "longitude"]).copy()
        lons = stn_df["longitude"].to_numpy(dtype=float)
        lats = stn_df["latitude"].to_numpy(dtype=float)
        
        utm_x, utm_y = self.to_utm.transform(lons, lats)
        stn_df["utm_x_km"] = utm_x / 1000.0  # Convert meters to kilometers
        stn_df["utm_y_km"] = utm_y / 1000.0

        self.stations_df = stn_df.reset_index(drop=True)
        self.coords_km = np.column_stack([self.stations_df["utm_x_km"], self.stations_df["utm_y_km"]])
        self.kdtree = KDTree(self.coords_km)
        logger.info("Initialized SpatialObservationIndexer with %d stations.", len(self.stations_df))

    def extract_grid_spatial_features(
        self,
        target_lat: float,
        target_lon: float,
        station_obs_dict: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculates all spatial features for a single target grid centroid:
        - Rejects any station with distance < 2 km
        - Computes aggregations across 2-5km, 5-10km, 10-20km, 20-50km
        - Computes pressure and humidity gradients
        """
        # Convert target coordinate to UTM kilometers
        x_m, y_m = self.to_utm.transform(target_lon, target_lat)
        t_x = x_m / 1000.0
        t_y = y_m / 1000.0

        # Query all stations within max radius (50 km)
        idx_within_50km = self.kdtree.query_ball_point([t_x, t_y], r=BAND_4_MAX_KM)
        
        if not idx_within_50km:
            return self._empty_feature_dict()

        # Compute exact distances and relative offsets (dx, dy)
        offsets = self.coords_km[idx_within_50km] - np.array([t_x, t_y])
        dists = np.linalg.norm(offsets, axis=1)

        # STRICT PROXIMITY EXCLUSION: Reject < 2.0 km
        valid_mask = dists >= EXCLUSION_RADIUS_KM
        if not np.any(valid_mask):
            return self._empty_feature_dict()

        valid_indices = [idx_within_50km[i] for i in range(len(dists)) if valid_mask[i]]
        valid_dists = dists[valid_mask]
        valid_offsets = offsets[valid_mask]

        # Station codes and observed values
        stn_codes = self.stations_df.iloc[valid_indices]["station_code"].tolist()
        
        rains = []
        pressures = []
        humidities = []
        keep_dists = []
        keep_offsets = []

        for code, d, off in zip(stn_codes, valid_dists, valid_offsets):
            obs = station_obs_dict.get(code, {})
            r = obs.get("hourly_rain", np.nan)
            p = obs.get("pressure", np.nan)
            h = obs.get("humidity", np.nan)
            
            rains.append(r)
            pressures.append(p)
            humidities.append(h)
            keep_dists.append(d)
            keep_offsets.append(off)

        rains = np.array(rains, dtype=float)
        pressures = np.array(pressures, dtype=float)
        humidities = np.array(humidities, dtype=float)
        keep_dists = np.array(keep_dists, dtype=float)
        keep_offsets = np.array(keep_offsets, dtype=float)

        features = {}

        # Band 1: 2 - 5 km
        b1_mask = keep_dists <= BAND_1_MAX_KM
        if np.any(b1_mask):
            r_b1 = rains[b1_mask]
            p_b1 = pressures[b1_mask]
            h_b1 = humidities[b1_mask]
            d_b1 = keep_dists[b1_mask]
            off_b1 = keep_offsets[b1_mask]

            features["rain_mean_2_5km"] = float(np.nanmean(r_b1)) if not np.all(np.isnan(r_b1)) else 0.0
            features["rain_max_2_5km"] = float(np.nanmax(r_b1)) if not np.all(np.isnan(r_b1)) else 0.0
            features["pressure_mean_2_5km"] = float(np.nanmean(p_b1)) if not np.all(np.isnan(p_b1)) else np.nan
            features["pressure_std_2_5km"] = float(np.nanstd(p_b1)) if not np.all(np.isnan(p_b1)) else 0.0
            features["humidity_mean_2_5km"] = float(np.nanmean(h_b1)) if not np.all(np.isnan(h_b1)) else np.nan
            features["humidity_std_2_5km"] = float(np.nanstd(h_b1)) if not np.all(np.isnan(h_b1)) else 0.0

            # Nearest station in Band 1
            min_idx = np.argmin(d_b1)
            features["nearest_stn_dist_2_5km"] = float(d_b1[min_idx])
            dx, dy = off_b1[min_idx]
            bearing = np.arctan2(dx, dy)
            features["nearest_stn_bearing_sin"] = float(np.sin(bearing))
            features["nearest_stn_bearing_cos"] = float(np.cos(bearing))
        else:
            features["rain_mean_2_5km"] = 0.0
            features["rain_max_2_5km"] = 0.0
            features["pressure_mean_2_5km"] = np.nan
            features["pressure_std_2_5km"] = 0.0
            features["humidity_mean_2_5km"] = np.nan
            features["humidity_std_2_5km"] = 0.0
            features["nearest_stn_dist_2_5km"] = 5.0
            features["nearest_stn_bearing_sin"] = 0.0
            features["nearest_stn_bearing_cos"] = 1.0

        # Band 2: 5 - 10 km
        b2_mask = (keep_dists > BAND_1_MAX_KM) & (keep_dists <= BAND_2_MAX_KM)
        if np.any(b2_mask):
            r_b2 = rains[b2_mask]
            p_b2 = pressures[b2_mask]
            h_b2 = humidities[b2_mask]
            features["rain_mean_5_10km"] = float(np.nanmean(r_b2)) if not np.all(np.isnan(r_b2)) else 0.0
            features["rain_max_5_10km"] = float(np.nanmax(r_b2)) if not np.all(np.isnan(r_b2)) else 0.0
            features["pressure_mean_5_10km"] = float(np.nanmean(p_b2)) if not np.all(np.isnan(p_b2)) else np.nan
            features["humidity_mean_5_10km"] = float(np.nanmean(h_b2)) if not np.all(np.isnan(h_b2)) else np.nan
        else:
            features["rain_mean_5_10km"] = 0.0
            features["rain_max_5_10km"] = 0.0
            features["pressure_mean_5_10km"] = np.nan
            features["humidity_mean_5_10km"] = np.nan

        # Band 3 & 4: 20 - 50 km
        b4_mask = (keep_dists > BAND_3_MAX_KM) & (keep_dists <= BAND_4_MAX_KM)
        if np.any(b4_mask):
            features["pressure_mean_20_50km"] = float(np.nanmean(pressures[b4_mask])) if not np.all(np.isnan(pressures[b4_mask])) else np.nan
            features["humidity_mean_20_50km"] = float(np.nanmean(humidities[b4_mask])) if not np.all(np.isnan(humidities[b4_mask])) else np.nan
        else:
            features["pressure_mean_20_50km"] = np.nan
            features["humidity_mean_20_50km"] = np.nan

        # Atmospheric Gradients (estimated using 2D plane fit / least squares over surrounding stations)
        p_grad_mag = self._compute_gradient_magnitude(keep_offsets, pressures)
        h_grad_mag = self._compute_gradient_magnitude(keep_offsets, humidities)
        features["pressure_gradient_mag"] = float(p_grad_mag)
        features["humidity_gradient_mag"] = float(h_grad_mag)

        return features

    @staticmethod
    def _compute_gradient_magnitude(offsets_km: np.ndarray, values: np.ndarray) -> float:
        """
        Computes spatial gradient magnitude ||grad V|| = sqrt((dV/dx)^2 + (dV/dy)^2)
        via 2D linear regression over surrounding observation points.
        """
        valid_idx = ~np.isnan(values)
        if np.sum(valid_idx) < 3:
            return 0.0

        x = offsets_km[valid_idx, 0]
        y = offsets_km[valid_idx, 1]
        v = values[valid_idx]

        # Fit plane: V = a*x + b*y + c
        A = np.column_stack([x, y, np.ones(len(x))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
            dv_dx, dv_dy = coeffs[0], coeffs[1]
            # Convert gradient to per 100 km unit
            grad_mag = np.sqrt(dv_dx ** 2 + dv_dy ** 2) * 100.0
            return float(np.clip(grad_mag, 0.0, 50.0))
        except Exception:
            return 0.0

    @staticmethod
    def _empty_feature_dict() -> Dict[str, float]:
        """Default feature values when no stations are present."""
        return {
            "rain_mean_2_5km": 0.0,
            "rain_max_2_5km": 0.0,
            "pressure_mean_2_5km": np.nan,
            "pressure_std_2_5km": 0.0,
            "humidity_mean_2_5km": np.nan,
            "humidity_std_2_5km": 0.0,
            "nearest_stn_dist_2_5km": 5.0,
            "nearest_stn_bearing_sin": 0.0,
            "nearest_stn_bearing_cos": 1.0,
            "rain_mean_5_10km": 0.0,
            "rain_max_5_10km": 0.0,
            "pressure_mean_5_10km": np.nan,
            "humidity_mean_5_10km": np.nan,
            "pressure_mean_20_50km": np.nan,
            "humidity_mean_20_50km": np.nan,
            "pressure_gradient_mag": 0.0,
            "humidity_gradient_mag": 0.0,
        }
