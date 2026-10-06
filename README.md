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
1. **แบบจำลองพยากรณ์อากาศโลก (NWP Forecasts)**: ข้อมูล ECMWF IFS จากฐานข้อมูล TIGGE
2. **ข้อมูลโทรมาตรสถานีวัดน้ำฝนภาคพื้นดิน (Ground Rain Gauges)**: จากสถาบันสารสนเทศทรัพยากรน้ำ (องค์การมหาชน) - HII
3. **ภาพถ่ายดาวเทียมอุตุนิยมวิทยาแบบวงโคจรค้างฟ้า (Geostationary Satellite)**: Himawari-9 Band 08 (Water Vapor) และ Band 13 (Clean IR)
4. **แบบจำลองระดับความสูงและภูมิประเทศเชิงพื้นที่ (Topography & Geomorphology)**: SRTM DEM, Slope, Aspect, Distance to Coast, และ Coriolis Parameter

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

### 3. ระดับการทดลองตัดทอนตัวแปร (3 Ablation Variants: 18 Runs + Baseline B0)
* **Baseline $B_0$**: ค่าพยากรณ์ฝนดิบจาก ECMWF IFS (Raw NWP)
* **$M_1$ (Ground Only)**: Raw NWP + ข้อมูลภูมิประเทศ + สถิติสถานีข้างเคียงภาคพื้นดิน
* **$M_2$ (Satellite Only)**: Raw NWP + ข้อมูลภูมิประเทศ + อุณหภูมิยอดเมฆและอัตราการก่อตัวจากดาวเทียม Himawari-9
* **$M_3$ (Full Multi-Modal)**: Raw NWP + ภูมิประเทศ + สถานีภาคพื้นดิน + ดาวเทียม Himawari-9

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
│   │   ├── himawari_extractor.py
│   │   └── tigge_extractor.py
│   ├── features/                       # Phase 3: Spatial Grid & 49 Feature Builder
│   │   ├── grid_generator.py           # 69,121 National Grid Cells (EPSG:4326 & EPSG:32647)
│   │   ├── spatial_features.py         # Multi-radius atmospheric gradients (<2km exclusion)
│   │   ├── feature_builder.py          # Unified 49-feature dataframe pipeline
│   │   └── test_spatial_features.py    # Unit tests ยืนยันความปลอดภัยเชิงพื้นที่
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

### 1. การเตรียมสภาพแวดล้อม (Environment Setup)
แนะนำให้ใช้งาน Python 3.10+ บน Virtual Environment หรือ Conda:

```bash
# Clone หรือเข้าสู่โฟลเดอร์ของโปรเจกต์
cd weather-forcast-enhance

# ติดตั้งแพ็กเกจที่จำเป็น
pip install numpy pandas scipy matplotlib lightgbm catboost torch pyproj pyarrow pyyaml
```

---

### 2. Phase 1 & 2: การตรวจสอบสถานีและป้องกันข้อมูลรั่วไหล (Data Ingestion & Audit)

ตรวจสอบความถูกต้องของพิกัดและแบ่งเกรดสถานี HII (Golden, Silver, Bronze):
```bash
# รันการวิเคราะห์พิกัดและคุณภาพข้อมูลสถานี HII
python src/data/hii_audit.py

# ตรวจสอบความถูกต้องของกระบวนการป้องกัน Data Leakage
python src/data/anti_leakage_validator.py
```

---

### 3. Phase 3: การสร้าง 2 km National Grid และฟีเจอร์เชิงพื้นที่ (Spatial Features Engine)

สร้างตารางกริด 2 km ทั่วประเทศ 69,121 เซลล์ และทดสอบการสร้างฟีเจอร์ 49 ตัวแปร:
```bash
# สร้าง 2 km Grid และคำนวณพิกัด UTM
python src/features/grid_generator.py

# รัน Unit Tests ตรวจสอบกฎการตัดสถานีในรัศมี < 2 km
python src/features/test_spatial_features.py

# รัน Feature Builder เพื่อสร้างตัวอย่างฟีเจอร์
python src/features/feature_builder.py
```

---

### 4. Phase 4: การรันและฝึกสอนโมเดล (Multi-NWP Model Training Protocol)

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

### 5. Phase 5: การประเมินผลและการสร้างชุดกราฟเส้นรายเดือน (Hydrological Evaluation & Charts)

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
  * Japan Meteorological Agency (JMA Himawari-9 Geostationary Satellite)
