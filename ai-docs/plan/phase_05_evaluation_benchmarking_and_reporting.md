# 📈 Phase 5: Hydrological Evaluation, Lead-Time Benchmark & RQ Validation

> **สถานะ**: ร่างแผนงาน (Planning Phase - Strict No Code)  
> **เป้าหมายหลัก**: ประเมินประสิทธิภาพของระบบพยากรณ์ฝนหลังปรับแก้ (Bias-Corrected Forecasts) บน 2 km Grid เทียบกับสถานีตรวจวัดน้ำฝนภาคพื้นดินระดับ Golden & Silver Tiers (จาก Phase 1) ตลอดทั้งปี 2025 โดยวิเคราะห์ทั้งมิติความคลาดเคลื่อนต่อเนื่อง (Continuous Metrics), ความแม่นยำในการเตือนภัยฝนตกหนัก (Categorical Rain Thresholds), การเสื่อมถอยตามระยะเวลานำหน้า (Lead-Time Horizon) และสรุปคำตอบสำหรับคำถามวิจัย 5 ข้อ (RQ1–RQ5)  
> **ข้อกำหนดพื้นที่ทำงาน**: โค้ดประเมินผล (`src/evaluation/`), ชุดข้อมูล Ground Truth (`data/clean_parquet/hourly_rain/`), และรายงานผลลัพธ์ (`outputs/`) ต้องอยู่ภายใต้ `weather-forcast-enhance/` ทั้งหมด ไม่มีการอ้างอิงไฟล์จากภายนอก

---

## 1. วิธีการจับคู่ประเมินผลเชิงพื้นที่ (Spatial Point-to-Grid Verification)

ตามข้อกำหนดใน [`weather-forcast-enhance/plan.md`](file:///e:/water-analysis-project/weather-forcast-enhance/plan.md) ข้อ 13:

```text
2 km × 2 km Grid Prediction (Lat, Lon)
                 │
                 ▼  (Nearest Neighbor หรือ Bilinear Interpolation)
        Interpolated Prediction
                 │
                 ▼  [ เปรียบเทียบ ณ Valid Time เดียวกัน ]
        Rain Gauge Observation
                 │
                 ▼
       Ground Truth (Golden Stations 2025)
```

- **การประเมิน 2 ระดับ (Two-Tier Ground Truth Evaluation Protocol)**:
  1. **Tier A: In-Network Benchmark (สถานี HII ปี 2025)**:
     - ใช้สถานีตรวจวัดน้ำฝน HII ระดับ Golden Tier ในปี 2025 เพื่อประเมินผลการพยากรณ์ข้ามปี (Temporal Out-of-Sample)
  2. **Tier B: Blind Out-of-Network Spatial Hold-Out (`weather-forcast-enhance/dataset/dwr_rain/`)** ⭐:
     - ใช้ข้อมูลฝนรายชั่วโมงสถานีตรวจวัดของ **กรมทรัพยากรน้ำ (DWR)** ที่จัดเก็บอยู่ใน `weather-forcast-enhance/dataset/dwr_rain/` (ครอบคลุมลุ่มน้ำสำคัญ เช่น ยม, น่าน, เจ้าพระยา, ชี, มูล, ป่าสัก, แม่กลอง, โขงเหนือ)
     - 🚨 **กฎเหล็กเด็ดขาด (Strict Invariant)**: ข้อมูลใน `dataset/dwr_rain/` **ห้ามนำไปใช้ในกระบวนการฝึกสอน (Train) หรือสร้างฟีเจอร์ใดๆ เด็ดขาด!**
     - เนื่องจากสถานี DWR เป็นเครือข่ายเซนเซอร์อิสระคนละหน่วยงานและมีตำแหน่งพิกัดต่างจาก HII อย่างสิ้นเชิง จึงเป็น **"สถานีที่ไม่รู้จัก 100% (Completely Blind & Unknown Spatial Hold-Out)"** โดยกำเนิด
     - การประเมินบนชุดสถานี DWR จะเป็นบทพิสูจน์ทางวิทยาศาสตร์ขั้นสูงสุดว่า โมเดลสามารถพยากรณ์ฝนบน 2 km Grid ในจุดที่ไม่เคยมีสถานีตรวจวัดของชุดฝึกได้อย่างแท้จริง (Zero Spatial Leakage / True Generalization)
- **การจับคู่เวลา**: เทียบผลพยากรณ์ ณ `valid_time` ตรงกับชั่วโมงตรวจวัดจริงของสถานี HII และ DWR

---

## 2. ดัชนีชี้วัดประสิทธิภาพ (Evaluation Metrics Suite)

### 2.1 ดัชนีวัดความต่อเนื่องเชิงปริมาณ (Continuous Quantitative Metrics)
1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (R_{\text{pred}, i} - R_{\text{obs}, i})^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |R_{\text{pred}, i} - R_{\text{obs}, i}|$$
3. **Mean Bias Error (MBE / Bias)**:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (R_{\text{pred}, i} - R_{\text{obs}, i})$$
   *(หากค่าเป็นบวกแสดงว่าพยากรณ์เกินจริง / Overestimation, ค่าเป็นลบแสดงว่าพยากรณ์ต่ำกว่าจริง / Underestimation)*
4. **Pearson Correlation ($r$) & Spearman Rank Correlation ($\rho$)**:
   - วัดความสอดคล้องเชิงความสัมพันธ์ของแนวโน้มการเพิ่ม-ลดของฝน

---

### 2.2 ดัชนีวัดการตรวจจับฝนตามระดับความรุนแรง (Rainfall Threshold & Contingency Metrics)
แบ่งเกณฑ์ระดับความรุนแรงของฝน 4 ระดับ:
- $\ge 0.1\text{ mm/h}$ (ฝนตก / Rain Occurrence)
- $\ge 2.0\text{ mm/h}$ (ฝนตกปานกลาง / Moderate Rain)
- $\ge 10.0\text{ mm/h}$ (ฝนตกหนัก / Heavy Rain - เริ่มเสี่ยงน้ำท่วมขัง)
- $\ge 20.0\text{ mm/h}$ (ฝนตกหนักมาก / Very Heavy Rain - เสี่ยงน้ำท่วมฉับพลัน)

สร้างตาราง Contingency Table 2x2 ณ แต่ละเกณฑ์ความรุนแรง:

| Forecast \ Observed | ฝนตกจริง ($\ge \tau$) | ไม่ตกจริง ($< \tau$) |
|---|:---:|:---:|
| **พยากรณ์ว่าตก ($\ge \tau$)** | Hits ($H$) | False Alarms ($F$) |
| **พยากรณ์ว่าไม่ตก ($< \tau$)** | Misses ($M$) | Correct Rejections ($C$) |

สูตรชี้วัดสำคัญทางอุตุนิยมวิทยาและการเตือนภัยพิบัติอุทกวิทยา:
1. **ความแม่นยำในการตรวจจับฝนหนัก (Heavy Rain Warning Accuracy / Hit Rate / POD)**:
   $$\text{POD (Probability of Detection)} = \frac{H}{H + M} \times 100\% \quad (\text{เป้าหมาย: } \to 100\%)$$
   - ชี้วัดว่า จากเหตุการณ์ฝนตกหนักจริงทั้งหมด ระบบสามารถออกประกาศเตือนได้สำเร็จกี่เปอร์เซ็นต์
2. **อัตราการเตือนหลอก (False Alarm Ratio: FAR - เตือนตกหนักแต่ดันไม่ตกจริง)**:
   $$\text{FAR} = \frac{F}{H + F} \times 100\% \quad (\text{เป้าหมาย: } \to 0\%)$$
   - ชี้วัดว่า ในบรรดาการร้องเตือนฝนตกหนักทั้งหมด มีกี่เปอร์เซ็นต์ที่เป็นการเตือนเก้อ (False Alarm เพื่อลดผลกระทบ Cry Wolf Effect)
3. **อัตราตกหนักแต่ไม่เตือน (Critical Miss Rate / False Negative - ความผิดพลาดอันตรายสูงสุด!)**:
   $$\text{Miss Rate} = \frac{M}{H + M} \times 100\% = 100\% - \text{POD} \quad (\text{เป้าหมาย: } \to 0\%)$$
   - ชี้วัดสัดส่วนที่ฝนตกหนักรุนแรง ($\ge 10\text{ mm/h}$ หรือ $\ge 20\text{ mm/h}$) แต่โมเดลพยากรณ์ว่าไม่ตก ซึ่งเสี่ยงต่อการเกิดน้ำท่วมฉับพลันโดยประชาชนไม่ได้เตรียมตัว
4. **ทักษะการเตือนภัยสุทธิ (Critical Success Index: CSI / Threat Score)**:
   $$\text{CSI} = \frac{H}{H + M + F} \quad (\text{เป้าหมาย: } \to 1.0)$$
   - เป็นดัชนีชี้วัดรวมที่ลงโทษทั้งการเตือนหลอก ($F$) และการไม่เตือน ($M$) พร้อมกัน
5. **F1-Score & Success Ratio (SR)**:
   $$\text{SR} = 1 - \text{FAR} = \frac{H}{H + F}, \quad \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 3. การวิเคราะห์ตาม Lead Time (Lead-Time Horizon Degradation)

ตามข้อกำหนดใน [`weather-forcast-enhance/plan.md`](file:///e:/water-analysis-project/weather-forcast-enhance/plan.md) ข้อ 14:

ผลการพยากรณ์จะถูกแจกแจงตามค่า `lead_time`:
$$\text{lead\_time} \in \{1\text{h}, 2\text{h}, 3\text{h}, \dots, 24\text{h}\}$$

```text
Error (RMSE mm/h)
 │
 │      Raw Weather Model (ECMWF IFS) ───- - - - - -
 │     /
 │    /   Bias-Corrected Model (M4) ────────────
 │   /
 │  /
 └──┴──────┴──────┴──────┴──────┴──────┴──────┴─────> Lead Time (Hours)
   +1h    +3h    +6h    +9h   +12h   +18h   +24h
```

### คำถามที่ต้องวิเคราะห์:
- ที่ Lead Time สั้น (+1h ถึง +3h) ข้อมูล Ground Observation และ Himawari-9 ช่วยลด Error ได้มากแค่ไหน?
- ที่ Lead Time ยาวขึ้น (+12h ถึง +24h) อิทธิพลของ Ground และ Satellite ค่อยๆ จางลงหรือไม่? และโมเดลปรับสมดุลอย่างไร?

---

## 4. แผนการตอบคำถามวิจัย (Research Questions Validation Matrix)

ระบบรายงานผลสรุปจะประเมินผลการทดลองทั้ง **18 Configurations ($6 \text{ Models} \times 3 \text{ Variants}: M_1, M_2, M_3$)** เทียบกับ **$B_0$ Baseline** เพื่อตอบโจทย์วิจัยทั้ง 5 ข้อใน `plan.md` ข้อ 15 อย่างลึกซึ้ง:

| รหัสคำถาม | คำถามวิจัย (Research Question) | การวิเคราะห์จากเมทริกซ์การทดลอง $Model \times 3$ |
|---|---|---|
| **RQ1** | การทำ ML bias correction สามารถลด error ของ Weather Model ในไทยได้หรือไม่? | เปรียบเทียบ **$B_0$ (Raw Model) เทียบกับ $M_3$ ของทุกโมเดล** (เช่น $B_0$ vs LGBM-M3, CAT-M3, HURDLE-M3) |
| **RQ2** | Spatial ground observations ช่วยปรับปรุง bias correction หรือไม่? | วิเคราะห์ผลต่าง **$M_1 - B_0$ ในทุกโมเดล** เพื่อดูว่าสถานีภาคพื้นดินช่วยดึง Skill Score ขึ้นเท่าใดในแต่ละอัลกอริทึม |
| **RQ3** | Himawari-9 ช่วยเพิ่มประสิทธิภาพเหนือ Weather Model + Ground หรือไม่? | วิเคราะห์ผลต่าง **$M_3 - M_1$ ในทุกโมเดล** เพื่อพิสูจน์ว่าภาพถ่ายดาวเทียมยอดเมฆช่วยลด Critical Misses ได้อย่างมีนัยสำคัญหรือไม่ |
| **RQ4** | ผลของ bias correction เปลี่ยนไปตาม lead time อย่างไร? | พล็อตกราฟ CSI และ RMSE แยกตาม Lead Time 1h–24h ของทั้ง 6 โมเดลในรอบ $M_3$ เพื่อดูการเสื่อมถอยตามระยะเวลานำหน้า |
| **RQ5** | ผลของ bias correction แตกต่างกันตามระดับความรุนแรงของฝนหรือไม่? | เปรียบเทียบความสามารถในการจับฝนหนัก ($\ge 10, \ge 20$ mm/h) ระหว่าง Single Tree (LGBM), Hurdle Model, และ Quantile ($q_{90}$) |

---

## 5. ผลลัพธ์และรายงานสรุป พร้อมตัวเลือก `--dir` (Deliverable Artifacts & CLI Specification)

สคริปต์ประเมินผลและการพล็อตกราฟ (`src/evaluation/run_evaluation.py` และ `src/evaluation/generate_benchmark_charts.py`) จะต้องรองรับ Argument `--dir` เพื่อให้ผู้ใช้สามารถระบุโฟลเดอร์สำหรับจัดเก็บ Model, ภาพกราฟ, ตารางสถิติ, และ CSV ผลการพยากรณ์ได้อย่างอิสระ:

```bash
# ตัวอย่างการสั่งรัน Evaluation และสร้างกราฟพร้อมระบุปลายทาง --dir
python src/evaluation/run_evaluation.py --dir outputs/benchmark_2025/
python src/evaluation/generate_benchmark_charts.py --dir outputs/benchmark_2025/

# หากไม่ระบุ --dir ระบบจะใช้โฟลเดอร์เริ่มต้น (Default): outputs/
python src/evaluation/run_evaluation.py
```

### โครงสร้างไฟล์ทั้งหมดที่บันทึกลงใน `--dir`:

```text
[output_dir]/                           # ไดเรกทอรีที่ระบุผ่าน --dir (เช่น outputs/benchmark_2025/)
├── models/                             # 1. จัดเก็บ Checkpoint / Weights ของแต่ละโมเดล
│   ├── lightgbm/                       # m1/, m2/, m3/
│   ├── catboost/                       # m1/, m2/, m3/
│   ├── hurdle/                         # m1/, m2/, m3/
│   ├── quantile/                       # m1/, m2/, m3/
│   ├── stgnn/                          # m1/, m2/, m3/
│   └── unet/                           # m1/, m2/, m3/
├── figures/                            # 2. จัดเก็บภาพกราฟความละเอียดสูงทั้งหมด (.png, .svg)
│   ├── 01_lead_time_rmse_mae_curves.png        # กราฟเส้น RMSE & MAE vs Lead Time (1h-24h) ทุกโมเดล
│   ├── 02_lead_time_csi_curves.png             # กราฟเส้น Threat Score (CSI) vs Lead Time (ฝนปกติและฝนหนัก)
│   ├── 03_model_vs_ablation_heatmap.png        # Heatmap เปรียบเทียบเมทริกซ์ 6 โมเดล x 3 Variants
│   ├── 04a_heavy_rain_warning_accuracy_pod.png # [สำคัญ] กราฟความแม่นยำเตือนฝนหนัก (Hit Rate / POD)
│   ├── 04b_heavy_rain_false_alarm_rate_far.png # [สำคัญ] กราฟอัตราเตือนหลอก (False Alarm Ratio: FAR)
│   ├── 04c_heavy_rain_critical_miss_analysis.png # [สำคัญ] กราฟตกหนักแต่ไม่เตือน (Critical Miss Rate & Count)
│   ├── 04d_performance_diagram_roebber.png     # Roebber Performance Diagram (POD vs 1-FAR vs CSI vs Bias)
│   ├── 05_dwr_blind_scatter_qq_plot.png        # Scatter & Q-Q Plot บนสถานี DWR (100% Blind Hold-Out)
│   ├── 06_basin_performance_barchart.png       # กราฟเปรียบเทียบ Skill Score แยกรายลุ่มน้ำสำคัญ
│   ├── 07_rain_intensity_radar_chart.png       # Radar Chart ตาม 4 ระดับความรุนแรงฝน
│   ├── 08_uncertainty_fan_chart.png            # Fan Chart แสดงช่วงความเชื่อมั่น q10-q90 ของพายุ
│   ├── 09_feature_importance_ranking.png       # จัดอันดับความสำคัญของฟีเจอร์ (SHAP Values)
│   └── 10_national_2km_bias_map.png            # แผนที่ 2 km Grid แสดงผลการพยากรณ์และปรับแก้ทั่วประเทศ
├── predictions/                        # 3. จัดเก็บ CSV Result ผลการพยากรณ์สำหรับใช้ใน Benchmark อื่นๆ
│   ├── unified_benchmark_predictions_2025.csv  # รวมผลพยากรณ์ทุกโมเดลและสถานีตลอดปี 2025
│   ├── dwr_blind_predictions_2025.csv          # เฉพาะผลพยากรณ์บนเครือข่ายสถานี DWR Hold-Out
│   └── <model>_<ablation>_pred.csv             # แยกรายโมเดล (เช่น lgbm_m3_pred.csv, hurdle_m3_pred.csv)
│       # คอลัมน์มาตรฐาน: valid_time, run_time, lead_time, station_id, lat, lon, basin_id,
│       #                 observed_rain, nwp_raw_rain, predicted_bias, corrected_rain, model_name
└── reports/                            # 4. สรุปรายงานตัวเลขและตารางสถิติ
    ├── final_benchmark_report_2025.md          # รายงานสรุปภาพรวมและคำตอบ RQ1–RQ5
    ├── lead_time_performance_breakdown.csv     # สถิติ error แยกรายชั่วโมง 1h-24h
    ├── rain_intensity_contingency_table.csv    # สถิติ CSI, POD, FAR แยกรายระดับฝน
    ├── model_ablation_matrix_18runs.csv        # สรุปผล 18 Runs (6 Models x 3 Variants) + B0
    ├── heavy_rain_warning_stats.csv            # สถิติ POD, FAR, Missed Count สำหรับฝน >=10 และ >=20 mm/h
    └── basin_skill_scores.csv                  # สถิติเปรียบเทียบแยกรายลุ่มน้ำ
```

---

## 6. สรุป Checklist ความพร้อม Phase 5

- [ ] ออกแบบฟังก์ชันคำนวณ Metrics ทั้งกลุ่ม Continuous (RMSE, MAE, Bias) และ Categorical (CSI, POD, FAR, F1)
- [ ] กำหนดกระบวนการจับคู่ Spatial Verification ระหว่าง 2 km Grid กับ Ground Truth Stations ปี 2025
- [ ] ออกแบบโครงสร้างตารางแจกแจงตาม Lead Time (+1h ถึง +24h)
- [ ] จัดวางเทมเพลตสำหรับเขียนสรุปผลงานวิจัยเพื่อตอบครบทั้ง 5 RQs (RQ1–RQ5)
