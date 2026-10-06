# 📊 HII Ground Data Zero-Null Completeness Audit Report (2021–2025)

> **Project**: Weather Model Precipitation Bias Correction (2 km × 2 km Grid)
> **Catalog Scope**: Hourly Rain, Air Pressure, Relative Humidity
> **Temporal Scope**: 2021-01-01 00:00:00 to 2025-12-31 23:00:00 (5 Years / 60 Months / 43,824 Hours)

---

## 1. Executive Summary

- **Total Catalog Master Stations**: 1,396 stations across Thailand
- **Stage 1 Stations with 100% Monthly Continuity (60/60 Months)**: N/A stations
- **Stage 2 Inspected Stations Count**: 15 stations
- **Tier 1 (Golden Zero-Null 100% Completeness)**: `0` station-catalog pairs
- **Tier 2 (Silver High-Quality >= 99.5%, gap <= 3h)**: `0` station-catalog pairs
- **Tier 3 (Bronze Operational 95.0% - 99.4%)**: `10` station-catalog pairs
- **Tier 4 (Excluded < 95.0%)**: `5` station-catalog pairs
- **Joint Triple-Golden Stations (Rain + Pressure + Humidity simultaneously 100%)**: `5` stations

---

## 2. Stage 1: Remote Presence Matrix Scan Results

| Catalog Variable | Total Stations Detected | Stations with Full 60 Months (100%) | 60-Month Continuity Rate |
|---|---|---|---|
| **hourly_rain** | 1,150 | 610 | 53.04% |
| **pressure** | 1,150 | 610 | 53.04% |
| **humidity** | 1,150 | 610 | 53.04% |

---

## 3. Station Quality Tier Breakdown (Stage 2 Deep Inspection)

| Quality Tier | Criteria | Count | Usage Recommendation |
|---|---|---|---|
| 🏆 **Tier 1: Golden** | 100% Valid (0 Null, 0 Sentinel, 0 Gap, 0 Flag Error) | **0** | **Ground Truth Target** for Phase 5 Evaluation & Core Training |
| 🥈 **Tier 2: Silver** | Completeness >= 99.5%, Max Gap <= 3 hours | **0** | Spatial Ground Observation with Spline Imputation (Flag: `INTERPOLATED`) |
| 🥉 **Tier 3: Bronze** | Completeness 95.0% – 99.4% | **10** | Secondary spatial density features |
| ❌ **Tier 4: Excluded** | Completeness < 95.0% | **5** | **Excluded** from training and evaluation |

---

## 4. Top Golden Stations (Sample)

*(No stations achieved strict 100% zero-null across the inspected subset)*

---

## 5. Joint Triple-Station Intersection (Rain + Pressure + Humidity)

Stations that possess **complete records for all 3 meteorological variables** concurrently at the same physical sensor location: **5** stations.

| station_code   | all_three_golden   | all_three_silver_or_better   |   min_completeness_pct |   avg_completeness_pct | hourly_rain_tier   |   hourly_rain_completeness_pct | pressure_tier    |   pressure_completeness_pct | humidity_tier    |   humidity_completeness_pct | station_name         | basin_name   | province_name   | amphoe_name   |   latitude |   longitude |
|:---------------|:-------------------|:-----------------------------|-----------------------:|-----------------------:|:-------------------|-------------------------------:|:-----------------|----------------------------:|:-----------------|----------------------------:|:---------------------|:-------------|:----------------|:--------------|-----------:|------------:|
| ACRU           | False              | False                        |                98.3959 |                98.8385 | Tier 3: Bronze     |                        99.7193 | Tier 3: Bronze   |                     98.4004 | Tier 3: Bronze   |                     98.3959 | ทต.ไก่คำ              | มูล           | อำนาจเจริญ       | เมืองอำนาจเจริญ |    15.7881 |     104.642 |
| ATG011         | False              | False                        |                62.1417 |                85.5726 | Tier 3: Bronze     |                        97.4192 | Tier 4: Excluded |                     62.1417 | Tier 3: Bronze   |                     97.1568 | ปตร.พลเทพ            | เจ้าพระยา     | ชัยนาท           | เมืองชัยนาท     |    15.2159 |     100.073 |
| ATG021         | False              | False                        |                39.9986 |                57.9811 | Tier 4: Excluded   |                        77.5465 | Tier 4: Excluded |                     39.9986 | Tier 4: Excluded |                     56.3983 | เหนือ ปตร.มโนรมย์      | เจ้าพระยา     | ชัยนาท           | มโนรมย์        |    15.3305 |     100.103 |
| ATG032         | False              | False                        |                50.2099 |                81.1633 | Tier 3: Bronze     |                        97.782  | Tier 4: Excluded |                     50.2099 | Tier 3: Bronze   |                     95.4979 | ท้ายปตร.บรมธาตุ        | เจ้าพระยา     | ชัยนาท           | เมืองชัยนาท     |    15.1578 |     100.153 |
| ATG042         | False              | False                        |                97.9212 |                97.9243 | Tier 3: Bronze     |                        97.9281 | Tier 3: Bronze   |                     97.9212 | Tier 3: Bronze   |                     97.9235 | ท้ายปตร.มะขามเฒ่า-อู่ทอง | ท่าจีน         | ชัยนาท           | วัดสิงห์         |    15.2223 |     100.062 |

---

## 6. Next Steps for Phase 2 & ML Pipeline

1. **Clean Parquet Archiving**: Run `hii_downloader.py` to persist verified clean records partitioned by catalog and year in `data/clean_parquet/`.
2. **Spatial Nearest Neighbor Matching**: For stations lacking co-located pressure/humidity sensors, use IDW (Inverse Distance Weighting, <= 5 km radius) to pair spatial pressure/humidity observations as specified in `plan.md`.
3. **Proceed to Phase 2**: Download Himawari-9 AHI IR/VIS satellite bands and NWP models (ECMWF IFS, AIFS, NCEP GEFS).
