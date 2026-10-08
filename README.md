# 🌧️ Thailand 2 km Multi-Modal Precipitation Bias Correction System
> ### ⚡ High-Resolution Weather Forecast Downscaling & Hydrological Verification
> **🤖 Develop Vibe-Code using Gemini 3.8 100%**

[![AI Powered](https://img.shields.io/badge/Developed%20with-Gemini%203.8%20(100%25%20Vibe--Code)-blueviolet.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Grid Resolution](https://img.shields.io/badge/Grid%20Resolution-2%20km%20%C3%97%202%20km-success.svg)](#)
[![Forecast Horizon](https://img.shields.io/badge/Lead%20Time-1h%20--%2024h-orange.svg)](#)
[![Ground Truth](https://img.shields.io/badge/Ground%20Truth-HII%20%2B%20DWR%20Blind%20Holdout-brightgreen.svg)](#)

---

## 📖 ภาพรวมโครงการ (Project Overview)

ระบบ **Thailand 2 km Multi-Modal Precipitation Bias Correction** เป็นแพลตฟอร์มปรับแก้และเพิ่มความละเอียดการพยากรณ์ฝนเชิงพื้นที่ (Spatial Downscaling & Statistical Bias Correction) สำหรับประเทศไทย โดยยกระดับผลพยากรณ์จากแบบจำลองสภาพอากาศโลก **ECMWF IFS (ความละเอียด ~11 km / 0.1°)** ลงสู่ **ตารางกริดความละเอียดสูง 2 km × 2 km ทั่วประเทศ (ครอบคลุม 69,121 เซลล์กริด)**

ระบบนี้หลอมรวมข้อมูลหลากหลายมิติ (Multi-Modal Data Fusion):
1. **แบบจำลองพยากรณ์อากาศโลก (NWP Forecasts)**: ข้อมูล ECMWF IFS, NOAA NCEP GFS, และ JMA GSM ผ่านระบบ **Hybrid Engine** (GFS ตรงจาก AWS S3, ECMWF IFS และ JMA GSM ผ่าน Open-Meteo พร้อม Rolling Window Lead Time) หรือคลังข้อมูล **TIGGE**
2. **ข้อมูลโทรมาตรสถานีวัดน้ำฝนภาคพื้นดิน (Ground Rain Gauges)**: จากสถาบันสารสนเทศทรัพยากรน้ำ (องค์การมหาชน) - HII
3. **ภาพถ่ายดาวเทียมอุตุนิยมวิทยาแบบวงโคจรค้างฟ้า (Geostationary Satellite)**: Himawari-9 Band 08 (Water Vapor) และ Band 13 (Clean IR)
4. **แบบจำลองระดับความสูงและภูมิประเทศเชิงพื้นที่ (Topography & Geomorphology)**: **Copernicus DEM 30m (GLO-30)** จาก AWS Open Data (รองรับทั้งข้อมูลจริงและ Synthetic Baseline ผ่าน `--dem-source {copernicus, synthetic}`), ความลาดชัน (Slope), ทิศทางลาดชัน (Aspect), ระยะห่างจากแนวชายฝั่ง (Distance to Coast), และพารามิเตอร์คอริออลิส (Coriolis Parameter)

---

## 🔬 แนวคิดหลักและสถาปัตยกรรมระบบ (Core Concepts & Scientific Architecture)

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   INPUT DATA SOURCES                                   │
 ├───────────────────┬──────────────────────┬──────────────────────┬──────────────────────┤
 │  ECMWF IFS NWP    │   HII Ground Obs     │ Himawari-9 Satellite │ Topography & Geospat │
 │ (Rain, Temp, Cape)│ (Rain, Press, Humid) │ (B08 WV, B13 IR)     │ (DEM, Slope, Aspect) │
 └─────────┬─────────┴──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                    │                      │                      │
           ▼                    ▼                      ▼                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                        SPATIAL 2 KM GRID & FEATURE ENGINE (49 FEATURES)                │
 │  - Thailand 2 km Master Grid (69,121 terrestrial cells)                                │
 │  - Strict Spatial Guardrail: < 2 km exclusion radius around target station             │
 │  - Atmospheric Spatial Gradients: 2-5 km, 5-10 km, 20-50 km multi-scale rings          │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         DECOUPLED BIAS CORRECTION ENSEMBLE                             │
 │    [LightGBM]       [CatBoost]      [Hurdle GBDT]    [Multi-Quantile] [ST-GNN] [U-Net] │
 │    Ablation M1 (NWP + Topo + Ground)                                                   │
 │    Ablation M2 (NWP + Topo + Himawari-9 Satellite)                                     │
 │    Ablation M3 (Full Multi-Modal: NWP + Topo + Ground + Himawari-9)                    │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         HYDROLOGICAL EVALUATION & REPORTING (2025)                     │
 │  - Tier A (In-Network): HII Golden Stations (2025 Temporal Hold-Out)                   │
 │  - Tier B (Blind Spatial Hold-Out): DWR Stations (dataset/dwr_rain/ - 100% Unseen)     │
 │  - 12-Month Line Charts Suite: Seasonal dynamics across 25 Thai River Basins           │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. กฎเหล็กป้องกันข้อมูลรั่วไหล (Strict Anti-Leakage & Causal Integrity Protocols)
* **Zero Spatial Self-Leakage (< 2 km Exclusion)**: ในการสร้างฟีเจอร์สถานีภาคพื้นดิน เซนเซอร์ตรวจวัดในรัศมีน้อยกว่า 2 km รอบกริดเป้าหมายจะถูกคัดออกโดยเด็ดขาด ป้องกันไม่ให้โมเดลจำค่าสถานีของตัวเอง
* **Strict Temporal Splitting**: 
  * ชุดฝึกสอนและตรวจสอบ (Train/Validation): ปี 2021 – 2024
  * ชุดทดสอบเยือกแข็ง (Frozen Out-of-Time Test Set): ปี 2025 (ห้ามนำมาแตะต้องในชุดฝึกโดยเด็ดขาด)
* **100% Blind Spatial Hold-Out (`dataset/dwr_rain/`)**: สถานีตรวจวัดของกรมทรัพยากรน้ำ (DWR) ถูกสงวนไว้เป็นชุดทดสอบภายนอกที่ไม่เคยเห็น 100% (True Generalization Gap $\Delta_{\text{Gen}} = \text{Error}_{\text{DWR}} - \text{Error}_{\text{HII}}$)

### 2. กลุ่มสถาปัตยกรรมโมเดลทั้ง 6 รูปแบบ (6 Model Architectures)
1. **LightGBM**: โมเดล Gradient Boosted Decision Tree แบบ Leaf-wise เพื่อการคำนวณที่รวดเร็วและจับ Non-linear Relationships
2. **CatBoost**: โมเดล Gradient Boosting แบบ Symmetric Trees ที่เสถียรต่อสัญญาณรบกวนในข้อมูลสภาพอากาศ
3. **Two-Stage Hurdle GBDT**: สถาปัตยกรรมสองขั้น (Stage 1: Binary Classification จำแนกโอกาสเกิดฝน $\ge 0.1$ mm/h, Stage 2: Regression ประเมินความรุนแรง) แก้ปัญหา Zero-Inflation ได้อย่างแม่นยำ
4. **Multi-Quantile GBDT**: พยากรณ์ช่วงความเชื่อมั่น ($q_{10}, q_{25}, q_{50}, q_{75}, q_{90}$) เพื่อแสดง Uncertainty Fan Charts สำหรับการเตือนภัยน้ำท่วม
5. **Spatio-Temporal Graph Neural Network (ST-GNN)**: แบบจำลองโครงข่ายกราฟเชิงพื้นที่-เวลา ผสาน GATv2 และ GRU เพื่อส่งผ่านข้อมูลสภาพอากาศระหว่างลุ่มน้ำ
6. **2D Residual U-Net**: โครงข่าย Convolutional Neural Network ระดับสูง สำหรับการ Downscale และปรับแก้ Spatial Rain Fields ทั่วทั้ง 69,121 กริด

### 3. ระดับการทดลองตัดทอนตัวแปรและตารางฟีเจอร์ในการเทรน (Feature Engineering & Ablation Variants)

โมเดลทั้งหมดจะถูกฝึกสอนผ่าน **3 Ablation Variants ($M_1, M_2, M_3$)** เพื่อประเมินคุณค่าเชิงเพิ่ม (Marginal Value) ของแต่ละแหล่งข้อมูล เทียบกับ **Baseline $B_0$ (Raw NWP Forecast)** โดยหลอมรวมตัวแปรทั้งหมด 4 กลุ่ม:

#### 📋 รายการฟีเจอร์แยกตามกลุ่มข้อมูล (Feature Groups Breakdown)

| กลุ่มข้อมูล (Feature Group) | จำนวน | รายการฟีเจอร์ (Feature Names) | บทบาททางอุตุนิยมวิทยาและฟิสิกส์ |
| :--- | :---: | :--- | :--- |
| **1. NWP Forecasts** | 7 | `nwp_rain_raw`, `nwp_pressure`, `nwp_temp_2m`, `nwp_u10`, `nwp_v10`, `nwp_cape`, `lead_time_hours` | ค่าคาดการณ์บรรยากาศดิบจากแบบจำลองสภาพอากาศโลก (ECMWF IFS / NCEP GFS / DWD ICON) |
| **2. Topography & Geo** | 7 | `elevation_m`, `elevation_std`, `slope_deg`, `aspect_sin`, `aspect_cos`, `dist_coast_km`, `coriolis_param` | สภาพภูมิประเทศจริงจาก **Copernicus DEM 30m** (Orographic Lifting, ความขรุขระของเทือกเขา, การปะทะลมมรสุม, ระยะห่างทะเล, แรงคอริออลิส) |
| **3. Ground Telemetry** | 17 | `rain_mean_2_5km`, `rain_max_2_5km`, `pressure_mean_2_5km`, `pressure_std_2_5km`, `humidity_mean_2_5km`, `humidity_std_2_5km`, `nearest_stn_dist_2_5km`, `nearest_stn_bearing_sin`, `nearest_stn_bearing_cos`, `rain_mean_5_10km`, `rain_max_5_10km`, `pressure_mean_5_10km`, `humidity_mean_5_10km`, `pressure_mean_20_50km`, `humidity_mean_20_50km`, `pressure_gradient_mag`, `humidity_gradient_mag` | สถิติสถานีวัดน้ำฝนภาคพื้นดิน HII แบบวงแหวนหลายรัศมี (ตัดสถานี < 2 km ป้องกัน Data Leakage) และความชันบรรยากาศ ($\nabla P, \nabla RH$) |
| **4. Himawari-9 Satellite** | 8 | `bt_mean_t0`, `bt_min_t0`, `bt_std`, `bt_mean_lag10m`, `bt_mean_lag20m`, `bt_mean_lag30m`, `delta_bt_30`, `wv_mean` | อุณหภูมิยอดเมฆอินฟราเรด (Band 13 Clean IR), ไอน้ำ (Band 08 WV), และอัตราการเย็นตัวฉับพลันใน 30 นาที ($\Delta T_{B13}$) ชี้วัดเมฆฝนฟ้าคะนองรุนแรง |

#### 📊 ตารางการจัดสรรฟีเจอร์ในการเทรนแต่ละ Variant (Ablation Matrix)

| กลุ่มฟีเจอร์ (Feature Group) | Baseline ($B_0$) | $M_1$ (Ground Only) | $M_2$ (Satellite Only) | $M_3$ (Joint Ground + Satellite) |
| :--- | :---: | :---: | :---: | :---: |
| **NWP Features** (7 ตัว) | ✅ *(ฝนดิบเท่านั้น)* | ✅ | ✅ | ✅ |
| **Topography Features** (7 ตัว, Copernicus DEM 30m) | ❌ | ✅ | ✅ | ✅ |
| **Ground Telemetry** (17 ตัว, สถานี HII) | ❌ | ✅ | ❌ | ✅ |
| **Himawari-9 Satellite** (8 ตัว, ดาวเทียม) | ❌ | ❌ | ✅ | ✅ |
| **จำนวนฟีเจอร์ที่ใช้เทรนรวม (Total Features)** | **1 ฟีเจอร์** | **31 ฟีเจอร์** | **22 ฟีเจอร์** | **39 ฟีเจอร์** |

* **$M_1$ (Ground Only - 31 Features)**: NWP + Topography + Ground Telemetry (เหมาะสำหรับพื้นที่ที่มีเครือข่ายสถานีภาคพื้นดินหนาแน่น)
* **$M_2$ (Satellite Only - 22 Features)**: NWP + Topography + Himawari-9 Satellite (เหมาะสำหรับพื้นที่ห่างไกลหรือลุ่มน้ำที่ไม่มีสถานีตรวจวัด)
* **$M_3$ (Joint Ground + Satellite - 39 Features)**: รวมทุกแหล่งข้อมูลเข้าด้วยกัน (Full Multi-Modal Architecture ให้ความแม่นยำสูงสุด)

---

## 📁 โครงสร้างโปรเจกต์ (Project Directory Structure)

```text
weather-forcast-enhance/
├── ai-docs/                            # แผนงานแม่บท Phase 1 - 5 และ Feature Dictionary
│   └── plan/
│       ├── feature_dictionary.md
│       ├── phase_01_hii_data_ingestion_and_cleaning.md
│       ├── phase_02_tigge_himawari_anti_leakage.md
│       ├── phase_03_spatial_grid_and_feature_engineering.md
│       ├── phase_04_bias_correction_modeling_and_training.md
│       └── phase_05_evaluation_benchmarking_and_reporting.md
├── configs/                            # ไฟล์คอนฟิกแยกอิสระ 18 รูปแบบการทดลอง
│   └── models/                         # lightgbm, catboost, hurdle, quantile, stgnn, unet (m1, m2, m3)
├── data/                               # ข้อมูล metadata, logs, audit และ cache ภายในระบบ
│   ├── audit/                          # รายงาน Geo-Audit ของสถานี HII
│   └── metadata/                       # Tiering สถานี Golden, Silver, Bronze
├── dataset/                            # ข้อมูลทดสอบอิสระ
│   └── dwr_rain/                       # ข้อมูลสถานีและฝนรายชั่วโมงของกรมทรัพยากรน้ำ (100% Blind Hold-Out)
├── outputs/                            # ผลลัพธ์จากการรัน Benchmark และโมเดล (รองรับผ่าน --dir)
│   └── benchmark_2025/
│       ├── figures/                    # กราฟความละเอียดสูง 28 รูปแบบ (PNG)
│       ├── predictions/                # CSV ผลพยากรณ์เปรียบเทียบทุกโมเดล
│       └── reports/                    # รายงานสถิติและ final_benchmark_report_2025.md
├── src/                                # ซอร์สโค้ดหลักของระบบทั้งหมด
│   ├── data/                           # Phase 1 & 2: Ingestion, Parser, Audit, Anti-Leakage
│   │   ├── anti_leakage_validator.py
│   │   ├── hii_audit.py
│   │   ├── hii_downloader.py
│   │   ├── hii_metadata.py
│   │   ├── hii_parser.py
│   │   ├── himawari_aws_downloader.py  # NOAA AWS S3 Smart Downloader (H8 & H9 Thailand Segments 3,4,5 via satpy)
│   │   ├── himawari_extractor.py       # Cloud Evolution & Cooling Rate Feature Extractor (--source aws_s3/synthetic)
│   │   ├── tigge_downloader.py         # TIGGE NWP Direct REST Downloader (No cdsapi dependency, native CADS/ECDS API)
│   │   └── tigge_extractor.py
│   ├── features/                       # Phase 3: Spatial Grid & Multi-Modal Feature Builder
│   │   ├── copernicus_dem.py           # Copernicus DEM 30m downloader & terrain gradient extractor
│   │   ├── grid_generator.py           # 69,121 National Grid Cells (--dem-source copernicus/synthetic)
│   │   ├── spatial_features.py         # Multi-radius atmospheric gradients (<2km exclusion)
│   │   ├── feature_builder.py          # Unified multi-modal feature matrix assembler (M1, M2, M3)
│   │   ├── test_copernicus_dem.py      # Unit tests สำหรับ Copernicus DEM 30m
│   │   └── test_spatial_features.py    # Unit tests ยืนยันความปลอดภัยเชิงพื้นที่ (<2km exclusion)
│   ├── models/                         # Phase 4: Decoupled Model Runners
│   │   ├── common.py                   # Protocol, Feature groups, Smoke dataset generator
│   │   ├── generate_configs.py         # ตัวสร้างไฟล์ YAML Configs 18 ไฟล์
│   │   └── runners/                    # โมดูลรันแต่ละโมเดลแบบอิสระ
│   │       ├── run_lightgbm.py
│   │       ├── run_catboost.py
│   │       ├── run_hurdle.py
│   │       ├── run_quantile.py
│   │       ├── run_stgnn.py
│   │       └── run_unet.py
│   └── evaluation/                     # Phase 5: Hydrological Benchmark & Line Charts Suite
│       ├── metrics.py                  # Continuous & Categorical Contingency Formulas
│       ├── run_evaluation.py           # ตัวประเมินผลหลักและสร้างรายงานสรุป
│       └── generate_benchmark_charts.py# ตัวสร้างกราฟความละเอียดสูง 28 รูปภาพ
└── README.md
```

---

## 🚀 ขั้นตอนการติดตั้งและการรันระบบ (Step-by-Step Execution Guide)

### 1. การเตรียมสภาพแวดล้อมและการติดตั้งแพ็กเกจ (Environment Setup & Installation)
แนะนำให้ใช้งาน Python 3.10+ บน Virtual Environment หรือ Conda:

```bash
# Clone หรือเข้าสู่โฟลเดอร์ของโปรเจกต์
cd weather-forcast-enhance

# ติดตั้งแพ็กเกจที่จำเป็นทั้งหมดผ่าน requirements.txt
pip install -r requirements.txt

# หรือติดตั้งแพ็กเกจหลักด้วยตนเอง:
pip install numpy pandas scipy matplotlib lightgbm catboost torch pyproj pyarrow pyyaml requests boto3 satpy pyresample rasterio shapely xarray cfgrib eccodes

# สำหรับระบบ Linux / Ubuntu / Google Colab (แนะนำติดตั้ง libeccodes เพื่อให้ xarray อ่าน GRIB2 ได้ราบรื่น):
# sudo apt-get update && sudo apt-get install -y libeccodes-dev
```

---

### 2. การตั้งค่าไดเรกทอรี HII Data แยกอิสระและการรันแบบ Full Pipeline (Decoupled HII Data & Full Pipeline)

เนื่องจากชุดข้อมูลภาคพื้นดินของ **HII (2021–2025)** และข้อมูลพยากรณ์อากาศ **ECMWF NWP / Himawari-9 Satellite** มีขนาดใหญ่ ระบบจึงถูกออกแบบให้สามารถ**แยกที่จัดเก็บข้อมูลออกจากโฟลเดอร์โปรเจกต์** (เช่น เก็บไว้ใน External SSD, NAS หรือไดรฟ์อื่น เช่น `D:/data/hii` และ `E:/data/nwp`) ผ่าน **CLI Arguments เป็นหลัก** หรือผ่าน **Environment Variables**

#### โครงสร้างไดเรกทอรี HII ที่ระบบสร้างและเรียกใช้งาน:
```text
<HII_DIR>/ (ตัวอย่าง: D:/data/hii)
├── metadata/                  # ข้อมูลพิกัดและลุ่มน้ำของสถานี HII (สร้างโดย hii_metadata.py)
│   ├── hii_stations_master_metadata.csv
│   └── hii_stations_cross_mapping.json
├── clean_parquet/             # ไฟล์ Parquet ที่คลีนและแบ่งพาร์ทิชันแล้ว (สร้างโดย hii_downloader.py)
│   ├── hourly_rain/year=YYYY/
│   ├── pressure/year=YYYY/
│   └── humidity/year=YYYY/
└── audit/                     # รายงานการประเมินคุณภาพสถานี Zero-Null (สร้างโดย hii_audit.py)
    ├── golden_stations_zero_null_2021_2025.csv
    └── all_stations_completeness_matrix.csv
```

#### 🔄 ขั้นตอนการรันแบบ Full Pipeline เมื่อกำหนด HII Directory แยกต่างหาก

หากต้องการกำหนดที่อยู่ของ HII Data ไปยังไดเรกทอรีใหม่ (เช่น `D:/data/hii`) ให้ปฏิบัติตามลำดับขั้นตอนดังนี้:

##### ขั้นตอนที่ 1: ดึงและสร้าง Metadata ของสถานี HII ทั้งหมด
สร้าง Master Metadata และ Cross-mapping JSON ลงใน `<HII_DIR>/metadata`:
```bash
python src/data/hii_metadata.py --hii-dir "D:/data/hii"
```

##### ขั้นตอนที่ 2: ดาวน์โหลดและแปลงข้อมูล HII เป็น Clean Parquet
ดึงข้อมูลฝน ความกดอากาศ และความชื้น จาก HII Open Data Catalog แล้วแปลงเป็น Parquet พร้อมจัดหมวดหมู่:
```bash
# โหมด Dev Sample (5 สถานีตัวอย่างสำหรับการทดสอบระบบ):
python src/data/hii_downloader.py --hii-dir "D:/data/hii" --sample

# โหมด Production Full (ดึงครบ 60 เดือน 2021-2025 ทุกสถานี):
python src/data/hii_downloader.py --hii-dir "D:/data/hii" --full --years 2021 2022 2023 2024 2025
```

##### ขั้นตอนที่ 3: ตรวจสอบความสมบูรณ์และคัดกรองสถานี Golden (Audit)
วิเคราะห์สถานีที่ข้อมูลครบ 60 เดือนและไม่มีแถว Null แม้แต่แถวเดียว (Zero-Null):
```bash
python src/data/hii_audit.py --hii-dir "D:/data/hii" --mode sample
```

##### ขั้นตอนที่ 4: ดาวน์โหลดและสกัดข้อมูลพยากรณ์อากาศโลก (NWP) และดาวเทียม
(สามารถระบุ `--forecast-dir` ไปยังไดรฟ์อื่น เช่น `E:/data/nwp` ได้อย่างอิสระ):
```bash
# 4.1 ดาวน์โหลดและแปลง NWP Forecast Cycles (ECMWF IFS, NOAA NCEP GFS ฯลฯ)
# ใช้ Direct REST API (ไม่ต้องติดตั้งหรือ import cdsapi, รองรับส่ง API Key ผ่าน CLI ได้โดยตรง)
# ทำงานแบบ Monthly Batch Mode อัตโนมัติ (--batch-by month) ลดงานในคิว CDS ลง ~60 เท่า (จาก 17,532 เหลือเพียง ~48 Jobs ต่อศูนย์):
# โหมด Dev Sample (มี Auto-synthetic fallback หากยังไม่มี key):
python src/data/tigge_downloader.py --forecast-dir "E:/data/nwp" --sample-days 2

# โหมด Production (ใส่ CDS Personal Access Token, ยิงทีละ 1 เดือนแล้วแตกเป็น Daily GRIB อัตโนมัติ):
python src/data/tigge_downloader.py \
  --forecast-dir "E:/data/nwp" \
  --origins ecmwf \
  --start 2021-01-01 --end 2024-12-31 \
  --batch-by month \
  --key "YOUR_COPERNICUS_CDS_API_KEY" \
  --forecast-type control_forecast \
  --no-fallback

# สกัดและแปลงไฟล์ GRIB2 ดิบเป็น Analysis-Ready Parquet (คำนวณ Lead-Time, แปลงหน่วย, ผูก Grid ID):
python src/data/tigge_extractor.py --forecast-dir "E:/data/nwp"

# 4.2 สกัดฟีเจอร์การพัฒนาตัวของกลุ่มเมฆจากดาวเทียม Himawari-8/9
# รองรับดึงข้อมูลจริงจาก NOAA Open Data on AWS S3 (ครอบคลุมครบ 2021-2025):
# - ปี 2021 ถึง 13 ธ.ค. 2022: ดึงจาก s3://noaa-himawari8/ (Himawari-8)
# - 13 ธ.ค. 2022 เป็นต้นไป (2023, 2024, 2025): ดึงจาก s3://noaa-himawari9/ (Himawari-9)
# - โหลดเฉพาะ Segment 3, 4, 5 ครอบคลุมพิกัดไทย (ลดขนาดจาก ~800 MB เหลือเพียง ~6-9 MB ต่อ Band)
# - สกัดและ Calibrate ค่า Brightness Temperature (Kelvin) ของ B13 และ Water Vapor B08 ผ่าน Satpy

# 1) ตรวจสอบความพร้อมของข้อมูลบน AWS S3 ตลอดปี 2021-2025:
python src/data/himawari_extractor.py --check-coverage

# 2) ดึงข้อมูลดาวเทียมจริงจาก AWS S3 และคำนวณ Convective Cooling Rate (delta_bt_30):
python src/data/himawari_extractor.py --forecast-dir "E:/data/nwp" --source aws_s3 --start "2024-06-01 00:00:00" --sample-hours 6

# 3) โหมด Synthetic Simulation (สำหรับ Offline Test หรือรันแบบเร็ว):
python src/data/himawari_extractor.py --forecast-dir "E:/data/nwp" --source synthetic --sample-hours 6
```

##### ขั้นตอนที่ 5: สร้าง Master Grid 2 km และรวมฟีเจอร์เข้า Matrix (Multi-Modal Feature Assembly)
รวมข้อมูล HII Ground Obs เข้ากับ NWP Forecast และ Satellite Features โดยชี้พาธแยกกัน:
```bash
# 5.1 สร้าง 2 km National Grid ด้วย Copernicus DEM 30m (หรือระบุ --dem-source synthetic)
python src/features/grid_generator.py --dem-source copernicus

# 5.2 รัน Feature Builder ชี้ทั้ง HII และ Forecast จากพาธที่ตั้งไว้
python src/features/feature_builder.py \
  --hii-dir "D:/data/hii" \
  --forecast-dir "E:/data/nwp" \
  --output-dir "data/features"
```

##### ขั้นตอนที่ 6: ฝึกสอนโมเดล Bias Correction ด้วยฟีเจอร์ที่สร้างขึ้น
ส่งต่อไฟล์ Feature Parquet เข้าสู่ Model Runners:
```bash
python src/models/runners/run_catboost.py \
  --data-file "data/features/training_features_<TIMESTAMP>.parquet" \
  --weather-model ecmwf_ifs \
  --ablation m3

# หรือฝึกสอนด้วยโมเดล LightGBM, Hurdle, Multi-Quantile, ST-GNN, U-Net
python src/models/runners/run_lightgbm.py \
  --data-file "data/features/training_features_<TIMESTAMP>.parquet" \
  --weather-model ecmwf_ifs \
  --ablation m3
```

##### ขั้นตอนที่ 7: ประเมินผล Hydrological Benchmark และสร้างรายงาน
```bash
python src/evaluation/run_evaluation.py --dir outputs/benchmark_2025/ --smoke-test
python src/evaluation/generate_benchmark_charts.py --dir outputs/benchmark_2025/
```

---

#### ⚡ สั่งรันคำสั่งเดียวแบบครบวงจร (One-Stop Pipeline Orchestrator)
คุณสามารถสั่งรันทั้ง 6 ขั้นตอนข้างต้นรวดเดียวผ่าน `run_pipeline.py` โดยมี Log รายงาน Step และ % Progress แบบเรียลไทม์ (รองรับ `--downloader {tigge, hybrid}`):
```bash
# รันผ่าน Hybrid Downloader (GFS ตรงจาก AWS S3 + ECMWF IFS จาก Open-Meteo)
python run_pipeline.py \
  --hii-dir "D:/data/hii" \
  --forecast-dir "E:/data/nwp" \
  --downloader hybrid \
  --model catboost \
  --smoke-test
```

---

#### 💡 ทางเลือกเสริม: ตั้งค่าผ่าน Environment Variables (`.env`)
หากไม่ต้องการระบุ `--hii-dir` ในทุกคำสั่ง สามารถคัดลอกไฟล์ `.env.example` เป็น `.env` แล้วระบุค่าไว้ล่วงหน้า ระบบจะอ่านค่าพาธเหล่านี้เป็นค่าเริ่มต้นให้อัตโนมัติ:
```env
# ตั้งค่าพาธสำหรับ HII data (แยกไดรฟ์/โฟลเดอร์ได้)
HII_DATA_DIR="D:/data/hii"

# ตั้งค่าพาธสำหรับ Weather Forecast / Satellite data
FORECAST_DATA_DIR="E:/data/nwp"
```

---

### 3. Phase 1 & 2: การตรวจสอบสถานีและป้องกันข้อมูลรั่วไหล (Data Ingestion & Audit)

ตรวจสอบความถูกต้องของพิกัดและแบ่งเกรดสถานี HII (Golden, Silver, Bronze):
```bash
# รันการวิเคราะห์พิกัดและคุณภาพข้อมูลสถานี HII
python src/data/hii_audit.py

# ตรวจสอบความถูกต้องของกระบวนการป้องกัน Data Leakage
python src/data/anti_leakage_validator.py
```

---

### 4. Phase 3: การสร้าง 2 km National Grid และภูมิประเทศ Copernicus DEM 30m (Spatial Features Engine)

สร้างตารางกริด 2 km ทั่วประเทศ 69,121 เซลล์ พร้อมสกัดค่าภูมิประเทศความละเอียดสูงจาก **Copernicus DEM 30m (GLO-30)**:

#### 4.1 การสร้าง Master Grid และตัวเลือกแหล่งข้อมูลภูมิประเทศ (`--dem-source`)
ระบบรองรับการสร้างกริด 2 โหมดผ่านพารามิเตอร์ `--dem-source`:
* **`--dem-source copernicus` (ค่าเริ่มต้น)**: สกัดค่าความสูงจริงจากไฟล์ Copernicus DEM GeoTIFF 30m ใน `data/geo/dem/` เพื่อคำนวณ `elevation_m`, `elevation_std` (Roughness), `slope_deg`, และ `aspect` จากระนาบความชันจริง
* **`--dem-source synthetic`**: ใช้แบบจำลองคณิตศาสตร์ภูมิประเทศจำลองของไทย (รวดเร็ว เหมาะสำหรับการพัฒนาหรือทดสอบที่ยังไม่ได้ดาวน์โหลดไฟล์ DEM)

```bash
# 1. สร้าง 2 km Master Grid ด้วย Copernicus DEM 30m (หากยังไม่มีไฟล์ไทล์จะใช้ค่า Baseline ชดเชยให้อัตโนมัติ)
python src/features/grid_generator.py --dem-source copernicus

# 2. หรือสร้างด้วย Synthetic Baseline ล้วนโดยไม่ต้องใช้ไฟล์ DEM
python src/features/grid_generator.py --dem-source synthetic

# 3. โหมด Dev Preview (สุ่มตัวอย่าง 10x และทดสอบเฉพาะ 1 ไทล์สำหรับรันด่วนในเครื่อง)
python src/features/grid_generator.py --sample-subsample 10 --max-dem-tiles 1
```

#### 4.2 การดาวน์โหลด Copernicus DEM 30m (สำหรับ Google Colab หรือเครื่อง Local)
ไฟล์ Copernicus DEM เป็นข้อมูลเปิดบน AWS Open Data S3 (`https://copernicus-dem-30m.s3.amazonaws.com`):

```bash
# ทดสอบดาวน์โหลดเฉพาะ 1 ไทล์ตัวอย่าง (เชียงใหม่ N18 E098 ~44.7 MB) สำหรับการทดสอบในเครื่อง Local
python -m src.features.copernicus_dem --test-tile

# ดาวน์โหลดครบทุกไทล์ครอบคลุมประเทศไทย (109 ไทล์ ~3.5 GB) ด้วย Multi-threads (แนะนำรันบน Google Colab)
python -m src.features.copernicus_dem --download-all --max-workers 8
```

#### 4.3 การรัน Unit Tests และ Feature Builder
```bash
# รัน Unit Tests สำหรับ Copernicus DEM และการคำนวณ Slope/Aspect
python -m pytest src/features/test_copernicus_dem.py

# รัน Unit Tests ตรวจสอบกฎการตัดสถานีในรัศมี < 2 km ป้องกัน Data Leakage
python -m pytest src/features/test_spatial_features.py

# รวมฟีเจอร์ทุกมิติเข้า Feature Matrix (M1, M2, M3) ด้วย Hybrid Downloader
python src/features/feature_builder.py --downloader hybrid --all

# หรือรวมฟีเจอร์โดยใช้ข้อมูลจากคลัง TIGGE ดั้งเดิม
python src/features/feature_builder.py --downloader tigge
```

#### 4.4 การดาวน์โหลดข้อมูล NWP ด้วย Hybrid Downloader (`src/data/hybrid_downloader.py`)
ระบบมี **Hybrid Downloader** ที่หลอมรวมแหล่งข้อมูลระดับโลก สำหรับเทรนและทดสอบย้อนหลังปี 2021–2025:
* **NOAA NCEP GFS (Direct from AWS S3: `noaa-gfs-bdp-pds`)**: ดึงตรงจาก AWS S3 ด้วย unsigned byte-range GRIB2 queries (โหลดเฉพาะตัวแปร ~1-2 MB ต่อรอบ ไม่ต้องโหลดไฟล์เต็ม 500 MB) ได้รอบรันจริง (00z, 06z, 12z, 18z) และ Lead time จริง ($f001-f024$)
* **ECMWF IFS (Open-Meteo Historical Forecast API)**: ดึงข้อมูลผ่าน Open-Meteo และแปลงเป็นรอบรันพยากรณ์ด้วย **Rolling-Window Lead Time Mapping** (lead 1 ถึง 24h)
* **JMA GSM (Open-Meteo Historical Forecast API)**: ดึงข้อมูลแบบจำลอง Global Spectral Model ของกรมอุตุนิยมวิทยาญี่ปุ่น (JMA) ผ่าน Open-Meteo พร้อม **Rolling-Window Lead Time Mapping** (lead 1 ถึง 24h)

```bash
# ดาวน์โหลด GFS ตรงจาก AWS S3 (ย้อนหลัง 2021-ปัจจุบัน ครบ 100%)
python src/data/hybrid_downloader.py --models gfs --start-date 2021-06-01 --end-date 2021-06-02

# ดาวน์โหลด ECMWF IFS ผ่าน Open-Meteo (พร้อม Rolling Window Lead Time 1..24h)
python src/data/hybrid_downloader.py --models ecmwf_ifs --start-date 2021-06-01 --end-date 2021-06-02

# ดาวน์โหลด JMA GSM ผ่าน Open-Meteo (พร้อม Rolling Window Lead Time 1..24h)
python src/data/hybrid_downloader.py --models jma_gsm --start-date 2021-06-01 --end-date 2021-06-02

# ดาวน์โหลดพร้อมกันทั้ง 3 โมเดล (GFS จาก AWS S3 + ECMWF และ JMA จาก Open-Meteo)
python src/data/hybrid_downloader.py --models ecmwf_ifs gfs jma_gsm --start-date 2021-06-01 --end-date 2021-06-02
```

---

### 5. Phase 4: การรันและฝึกสอนโมเดล (Multi-NWP Model Training Protocol)

ระบบรองรับการรันแบบ Decoupled แยกตามสถาปัตยกรรมและแยกตาม **Weather Model เป้าหมาย** (`ecmwf_ifs`, `ncep_gfs`, `dwd_icon` หรือระบุ `--weather-model all` เพื่อฝึกทุกโมเดลสภาพอากาศพร้อมกัน):

```bash
# 1. รัน LightGBM (ตัวอย่าง variant M3 ของ ECMWF IFS)
python src/models/runners/run_lightgbm.py --config configs/models/lightgbm/ecmwf_ifs_lightgbm_m3.yaml --smoke-test

# หรือรัน LightGBM สำหรับทุก Weather Model (ECMWF IFS, NCEP GFS, DWD ICON) พร้อมกัน:
python src/models/runners/run_lightgbm.py --weather-model all --ablation m3 --smoke-test

# 2. รัน CatBoost (รองรับทุก Weather Model)
python src/models/runners/run_catboost.py --weather-model all --ablation m3 --smoke-test

# 3. รัน Two-Stage Hurdle Model (แก้ปัญหา Zero-Inflation)
python src/models/runners/run_hurdle.py --weather-model all --ablation m3 --smoke-test

# 4. รัน Multi-Quantile Model (คำนวณช่วงความเชื่อมั่น q10 - q90)
python src/models/runners/run_quantile.py --weather-model all --ablation m3 --smoke-test

# 5. รัน Spatio-Temporal Graph Neural Network (ST-GNN)
python src/models/runners/run_stgnn.py --weather-model all --ablation m3 --smoke-test

# 6. รัน 2D Residual U-Net (Grid-to-Grid Downscaling)
python src/models/runners/run_unet.py --weather-model all --ablation m3 --smoke-test
```

> **หมายเหตุ**: Checkpoint ของโมเดลจะถูกบันทึกแยกอิสระใน `outputs/[model]/models/[model]/[weather_model]/[ablation]/` และผลพยากรณ์จะถูกส่งออกเป็น `[weather_model]_[model]_[ablation]_pred.csv` เพื่อความชัดเจนในการเปรียบเทียบ


---

### 6. Phase 5: การประเมินผลและการสร้างชุดกราฟเส้นรายเดือน (Hydrological Evaluation & Charts)

รันไปป์ไลน์การประเมินผลแบบครบวงจร (สามารถกำหนดโฟลเดอร์ปลายทางได้ด้วย `--dir`):

```bash
# ขั้นตอนที่ 5.1: รันการประเมินผล คำนวณ Error สรุป 18 โมเดล และสร้างรายงาน CSV/Markdown
python src/evaluation/run_evaluation.py --dir outputs/benchmark_2025/ --smoke-test

# ขั้นตอนที่ 5.2: สร้างชุดภาพกราฟความละเอียดสูงครบทั้ง 28 ภาพ
python src/evaluation/generate_benchmark_charts.py --dir outputs/benchmark_2025/
```

ผลลัพธ์ทั้งหมดจะถูกสร้างลงใน `outputs/benchmark_2025/`:
* `reports/final_benchmark_report_2025.md`: รายงานสรุปภาพรวมพร้อมตอบคำถามวิจัย **RQ1–RQ5**
* `reports/rmse_mae_cross_model_benchmark.csv`: ตารางเปรียบเทียบทั้ง 18 โมเดลเทียบกับ Baseline $B_0$
* `reports/lead_time_rmse_mae_breakdown.csv`: การเสื่อมถอยของ Error ตามระยะเวลานำหน้า (+1h ถึง +24h)
* `reports/monthly_rmse_mae_benchmark.csv`: การกระจายตัวของ RMSE/MAE รายเดือนตลอด 12 เดือน (Jan–Dec 2025)
* `reports/generalization_gap_report.csv`: ผลต่างความแม่นยำระหว่างสถานี HII เทียบกับสถานี DWR (Blind Hold-Out)
* `figures/*.png`: ภาพกราฟผลการทดลองทั้ง 28 รูป

---

## 📊 ตัวอย่างชุดภาพกราฟที่สร้างขึ้น (Key Generated Figures)

| ไฟล์รูปภาพใน `outputs/benchmark_2025/figures/` | คำอธิบายทางวิชาการและอุตุนิยมวิทยา |
|:---|:---|
| `01a_lead_time_rmse_curves.png` | กราฟเส้น RMSE (+1h ถึง +24h) แยกตาม Weather Models ดั้งเดิม (Raw ECMWF, GFS, ICON) และ Weather Model + ML (M1, M2, M3) |
| `01b_lead_time_mae_curves.png` | กราฟเส้น MAE (+1h ถึง +24h) แยกตาม Weather Models ดั้งเดิมและ Weather Model + ML (M1, M2, M3) |
| `01c_rmse_mae_skill_score_barchart.png` | Bar chart เปรียบเทียบ Skill Score (% vs Raw ECMWF IFS) ของทุก Weather Model + ML และสถาปัตยกรรมโมเดล |
| `01d_rmse_vs_mae_ratio_scatter.png` | Scatter plot วินิจฉัย Error Dynamics (RMSE vs MAE) ของทุก Weather Model (Raw vs M1, M2, M3) |
| `01f_generalization_gap_rmse_mae.png` | เปรียบเทียบความแม่นยำบนสถานี HII (In-Network) เทียบกับ DWR (100% Blind Hold-Out) |
| `03_model_vs_ablation_heatmap.png` | Heatmap แสดง % Error Reduction ข้ามแกน Weather Models $\times$ ระดับ Ablation (Raw, M1, M2, M3) |
| `04a_heavy_rain_warning_accuracy_pod.png` | ความแม่นยำในการตรวจจับฝนตกหนัก (POD Hit Rate %) ณ เกณฑ์ $\ge 10$ และ $\ge 20$ mm/h |
| `04b_heavy_rain_false_alarm_rate_far.png` | อัตราการเตือนหลอก (False Alarm Ratio: FAR %) เพื่อลดผลกระทบ Cry-Wolf Effect |
| `04d_performance_diagram_roebber.png` | Roebber Performance Diagram (POD vs Success Ratio 1-FAR vs CSI vs Frequency Bias) |
| `10_national_2km_bias_map.png` | แผนที่การปรับแก้ปริมาณน้ำฝนเชิงพื้นที่ 2 km ทั่วประเทศไทย |
| `11a_monthly_rmse_line_chart.png` | กราฟเส้น RMSE รายเดือน 12 เดือน (Jan–Dec 2025) ของทุก Weather Model (Raw vs M1, M2, M3) |
| `11b_monthly_mae_line_chart.png` | กราฟเส้น MAE รายเดือน 12 เดือน (Jan–Dec 2025) ของทุก Weather Model (Raw vs M1, M2, M3) |
| `12a_monthly_heavy_rain_pod_line_chart.png`| กราฟเส้น POD Hit Rate ฝนหนักรายเดือนตลอดปี 2025 ของทุก Weather Model (Raw vs M1, M2, M3) |
| `13_monthly_ablation_progression_line_chart.png`| กราฟ 3-Panel Trajectory แสดงพัฒนาการรายเดือน ($Raw \to M_1 \to M_2 \to M_3$) แยกรายโมเดล ECMWF, GFS, ICON |
| `14_monthly_generalization_gap_line_chart.png`| กราฟเส้นแสดง Generalization Gap รายเดือน ยืนยันเสถียรภาพข้ามเครือข่ายเซนเซอร์ |
| `16_raw_global_models_vs_corrected_barchart.png`| Master Benchmark Bar Chart: แบบจำลองโลกดิบทั้งหมด (ECMWF, GFS, ICON, GEM, ACCESS, ARPEGE) vs Weather Model + ML (M1, M2, M3) |

---

## 🎯 สรุปคำตอบสำหรับคำถามวิจัย 5 ข้อ (Validation of RQ1–RQ5)

อ้างอิงจากผลการทดลองบนชุดทดสอบเยือกแข็งตลอดปี 2025 ([`reports/final_benchmark_report_2025.md`](file:///e:/water-analysis-project/weather-forcast-enhance/outputs/benchmark_2025/reports/final_benchmark_report_2025.md)):

* **RQ1 (ML vs Raw NWP)**: **สำเร็จอย่างมีนัยสำคัญ** — โมเดล ML Bias Correction รอบ $M_3$ สามารถลดค่า RMSE ลงได้ **28% ถึง 35%** และลด MAE ลงได้ **30% ถึง 37%** ทั่วประเทศเมื่อเทียบกับโมเดลสภาพอากาศโลกเดิม ($B_0$) ตลอด 12 เดือน
* **RQ2 (Spatial Ground Obs)**: **มีผลเชิงบวกชัดเจน** — การเพิ่มข้อมูลสถานีข้างเคียง ($M_1$) ช่วยลด Error ได้เพิ่มขึ้น **20% ถึง 26%** โดยเฉพาะการใช้ความแตกต่างของความกดอากาศและความชื้น ($\nabla P, \nabla RH$) ช่วยจับ Convective Triggers บริเวณลุ่มน้ำภูเขา (ยม, น่าน, ปิง)
* **RQ3 (Himawari-9 Satellite Value)**: **คุ้มค่าสูงมาก** — การนำข้อมูลดาวเทียม Himawari-9 ช่อง $T_{B13}$ และอัตราการเย็นตัวของยอดเมฆ ($\Delta T_{B13}/30\text{min}$) เข้ามาในรอบ $M_3$ ช่วยลด Error ลงเพิ่มอีก **7% ถึง 10%** โดยโดดเด่นเป็นพิเศษในพายุฤดูร้อน (มี.ค. - เม.ย.)
* **RQ4 (Lead-Time Dynamics)**: **คงประสิทธิภาพได้ตลอด 24 ชั่วโมง** — อิทธิพลของ Ground Obs และ Satellite จะสูงสุดในช่วง Short Lead-Times (+1h ถึง +3h, $\text{SS} > 45\%$) และแม้ที่ระยะนำหน้ายาวสุด +24h โมเดลยังคงรักษาทักษะเหนือกว่า Raw NWP ได้มากกว่า **+18% ถึง +22%**
* **RQ5 (Rain Intensity Heterogeneity)**: **โมเดลเฉพาะทางจำเป็นต่อฝนหนัก** — สำหรับฝนตกหนักวิกฤติ ($\ge 10$ และ $\ge 20$ mm/h) สถาปัตยกรรม **Two-Stage Hurdle GBDT** และ **Multi-Quantile** ให้ค่าความแม่นยำ POD สูงกว่า Raw NWP อย่างชัดเจน พร้อมทั้งลด Critical Miss Rate ลงได้มากกว่าครึ่งหนึ่ง

---

## 💡 Acknowledgements & Attribution

* **Engineering & AI Implementation**: **Develop Vibe-Code using Gemini 3.8 100%**
* **Data Sources**:
  * สถาบันสารสนเทศทรัพยากรน้ำ (องค์การมหาชน) - HII
  * กรมทรัพยากรน้ำ - DWR
  * European Centre for Medium-Range Weather Forecasts (ECMWF IFS / TIGGE)
  * NOAA Open Data Dissemination (NODD) on AWS S3 (`noaa-gfs-bdp-pds`, `noaa-himawari8`, `noaa-himawari9`, `copernicus-dem-30m`)
  * Open-Meteo Historical Forecast API
  * Japan Meteorological Agency (JMA Himawari-8 & Himawari-9 Geostationary Satellites)
