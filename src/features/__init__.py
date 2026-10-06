"""
Spatial Grid Generation & Meteorological Feature Engineering Module
Supports:
- Thailand 2 km x 2 km Master Grid generation (EPSG:32647 UTM 47N / EPSG:4326)
- Topographic attribute calculation (DEM elevation, slope, aspect, coast distance, Coriolis)
- KDTree spatial indexing with strict proximity exclusion (< 2 km)
- Distance band spatial aggregations (2-5 km, 5-10 km, 10-20 km, 20-50 km)
- Atmospheric spatial gradients (Pressure gradient, Humidity gradient)
- Unified multi-modal feature matrix assembler (NWP + Satellite + Ground + Topo)
"""

__version__ = "1.0.0"
