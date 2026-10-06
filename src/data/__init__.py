"""
HII Ground Data Acquisition and Audit Module
Supports:
- Metadata harmonization across HII Open Data Catalogs (Rain, Pressure, Humidity)
- Dual-era CSV parsing (2021-2024 legacy vs 2025 modern schema)
- Zero-null & data completeness auditing (Golden, Silver, Bronze, Excluded tiers)
- Harmonized Parquet export for downstream NWP ML bias correction
"""

__version__ = "1.0.0"
