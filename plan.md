## 1. เป้าหมาย

สร้างระบบ **Weather model precipitation bias correction สำหรับประเทศไทย** ที่สามารถสร้าง rainfall forecast บน **2 km × 2 km grid**

```text
Weather model (ECMWF IFS , ECMWF AIFS, ECMWF AI-AIFS ,NCEP GEFS, NCEP AI-GEFS , Google WeatherNext 2) Forecast
      +
Himawari-9
      +
Spatial Ground Observations
      +
Geographic Features
      ↓
ML Bias Correction
      ↓
Rainfall Forecast
      ↓
2 km × 2 km Grid ทั่วประเทศไทย
```

สถานีฝน **ไม่ใช่ target ของระบบ**

แต่ใช้เป็น:

* ground observation สำหรับสร้าง spatial features
* ground truth สำหรับประเมินผล

---

# 2. Forecast cycle ต้องเป็นแกนหลัก

เนื่องจาก ECMWF forecast ถูกออกเป็นหลาย **forecast runs ต่อวัน** เราจะเก็บข้อมูลในรูปแบบ:

```text
forecast_run_time
forecast_valid_time
```

แล้วคำนวณ:

```text
lead_time
=
forecast_valid_time
-
forecast_run_time
```

ตัวอย่าง:

```text
Run: 2025-01-01 00:00
        │
        ├── Valid 01:00 → lead 1h
        ├── Valid 02:00 → lead 2h
        ├── Valid 03:00 → lead 3h
        ├── ...
        └── Valid 24:00 → lead 24h
```

ดังนั้นเรา **ไม่ต้องสมมติว่า +3h เป็น target หลักตั้งแต่แรก**

---

# 3. Dataset

แต่ละ row จะหมายถึง:

> **Forecast ณ run หนึ่ง สำหรับ target grid หนึ่ง ณ valid time หนึ่ง**

เช่น:

```text
run_time
valid_time
lead_time

target_lat
target_lon

ecmwf_precipitation
ecmwf_temperature
ecmwf_pressure
...

ground_features
himawari_features
...
```

ตัวอย่าง:

```text
run_time      valid_time       lead
00:00         03:00            3h
00:00         04:00            4h
00:00         05:00            5h

06:00         09:00            3h
06:00         10:00            4h
...
```

ตรงนี้สำคัญมาก เพราะจะทำให้ dataset ของคุณ **มีความหมายทาง meteorological forecast จริง ๆ**

---

# 4. Target

ยังคงใช้ **2 km × 2 km grid**

เช่น:

```text
Grid A
lat = ...
lon = ...

Grid B
lat = ...
lon = ...
```

ไม่มีข้อจำกัดว่าต้องมี rain station อยู่ใน grid

---

# 5. Ground station

ใช้ station เป็น spatial observation

สำหรับ target grid:

```text
        Station
           ●
          /
         / distance
        /
       ★ Target Grid
```

เก็บ:

```text
rain
pressure
humidity
distance_km
bearing
```

และแปลง bearing:

```text
bearing_sin
bearing_cos
```

---

# 6. ระยะห่างสถานี

กำหนด:

```text
< 2 km
→ ไม่ใช้เป็น nearby station
```

เพราะเราไม่อยากให้ target grid ที่มี station อยู่ตรงนั้นสามารถเห็น observation ที่แทบจะเป็นจุดเดียวกัน

ใช้:

```text
2–5 km
5–10 km
10–20 km
20–50 km
```

และให้ model เห็น `distance_km` ด้วย

---

# 7. Spatial Ground Features

เริ่มจาก:

```text
pressure
humidity
rain
distance
bearing_sin
bearing_cos
```

แล้ว aggregate เช่น:

```text
pressure_mean_2_5km
pressure_std_2_5km

pressure_mean_5_10km

humidity_mean_2_5km

rain_mean_2_5km
rain_max_2_5km
```

และภายหลังอาจเพิ่ม:

```text
pressure_gradient
humidity_gradient
rainfall_gradient
```

---

# 8. Himawari-9

ใช้เพื่อให้ model เห็น **cloud state และ cloud evolution**

Feature เช่น:

```text
BT_mean
BT_std
BT_min
BT_max
```

รอบ target grid

รวม temporal features:

```text
BT(t)
BT(t-10m)
BT(t-20m)
BT(t-30m)

ΔBT
```

โดยต้อง enforce กฎ:

> ณ `run_time` ห้ามใช้ข้อมูล Himawari ที่เกิดหลัง `run_time`

เพื่อป้องกัน leakage

---

# 9. Bias Correction Target

ยังใช้แนวคิด residual:

```text
bias =
observed_rainfall
-
ECMWF_rainfall
```

Model เรียน:

```text
Predicted Bias
```

แล้ว:

```text
Corrected Rainfall
=
Weather model Rainfall
+
Predicted Bias
```

นี่ทำให้งานยังเป็น **Weather model Bias Correction** อย่างชัดเจน

---

# 10. Model

Baseline:

```text
B0 = ECMWF raw
```

จากนั้นทำ ablation:

```text
M1 = Weather model + Ground

M2 = Weather model + Himawari

M3 = Weather model + Ground + Himawari
```

และอาจเพิ่ม:

```text
M4 = ECMWF + Ground + Himawari
     + Spatial features
```

Model หลัก:

**LightGBM**

โดยอาจ benchmark กับ Ridge/XGBoost/CatBoost

---

# 11. Time Split

ใช้ข้อมูล:

```text
2021
2022
2023
2024
```

สำหรับ train/validation

และ:

```text
2025
```

เป็น **final test**

```text
2021 ─────── 2022 ─────── 2023 ─────── 2024
                  TRAIN / VALIDATION

2025
──────────────────────────────────────────
                 FINAL TEST
```

ห้ามใช้ 2025 เลือก feature หรือ hyperparameter

---

# 12. Validation

ภายใน 2021–2024 ใช้ temporal validation

เช่น:

```text
Train       Validation

2021–2022 → 2023
2021–2023 → 2024
```

จากนั้น:

```text
Train final = 2021–2024

Test = 2025
```

---

# 13. Evaluation

ในแต่ละ `valid_time` เอา prediction ของ grid ที่อยู่ใกล้ station ไปเทียบกับ station observation

```text
2 km Grid prediction
        ↓
     Prediction
        ↓
     Compare
        ↓
Rain gauge observation
        ↓
   Ground Truth
```

Metrics:

### Continuous

```text
RMSE
MAE
Bias
Correlation
```

### Rain threshold

เช่น:

```text
≥ 2 mm
≥ 10 mm
≥ 20 mm
```

วัด:

```text
POD
FAR
CSI
Precision
Recall
F1
```

---

# 14. Lead time

ตรงนี้เปลี่ยนจากแผนเดิมครับ

**ไม่กำหนดว่า +3/+6/+9/+12 เป็น target หลักตั้งแต่ต้น**

แต่ให้เก็บ `lead_time` จาก:

```text
valid_time - run_time
```

แล้ววิเคราะห์ performance ตาม lead time ที่ dataset มีจริง

เช่นผลสุดท้ายอาจออกมาเป็น:

```text
Lead time     ECMWF       Corrected
1h            ...          ...
2h            ...          ...
3h            ...          ...
4h            ...          ...
...
24h           ...          ...
```

ถ้า Open-Meteo historical data ที่คุณใช้มี forecast horizon ถึง 7 วัน ก็สามารถดูได้ไกลกว่านั้น แต่ **ไม่จำเป็นต้องเอาทุก lead time มาเป็น scope หลักของ thesis**

---

# 15. Research Questions

งานจะตอบประมาณนี้:

### RQ1

**การทำ ML bias correction สามารถลด error ของ ECMWF precipitation forecast ในประเทศไทยได้หรือไม่?**

### RQ2

**Spatial ground observations ช่วยปรับปรุง bias correction หรือไม่?**

### RQ3

**Himawari-9 ช่วยเพิ่มประสิทธิภาพเหนือ ECMWF + Ground หรือไม่?**

### RQ4

**ผลของ bias correction เปลี่ยนไปตาม lead time อย่างไร?**

### RQ5

**ผลของ bias correction แตกต่างกันตามระดับความรุนแรงของฝนหรือไม่?**

---

# Architecture สุดท้าย

```text
              Weather model Forecast
                    │
                    │
              Forecast Run
                    │
             ┌──────┴──────┐
             │             │
        Himawari-9     Ground Stations
             │             │
             │       ┌─────┴─────┐
             │       │           │
             │    Pressure    Humidity
             │    Rainfall    Rainfall
             │       │           │
             │   Distance + Bearing
             │       │
             └───────┴───────────┘
                     ↓
                Feature Set
                     ↓
                  LightGBM
                     ↓
             Predicted Bias
                     ↓
          ECMWF + Predicted Bias
                     ↓
              2 km Grid
                     ↓
          Nationwide Forecast
                     ↓
              Rain Stations
                     ↓
             Ground Truth
```