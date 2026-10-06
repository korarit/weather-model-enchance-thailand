# 🛰️ Phase 2: Remote Sensing (Himawari-9) & NWP Forecast Cycles Integration

> **สถานะ**: ร่างแผนงาน (Planning Phase - Strict No Code)  
> **เป้าหมายหลัก**: วางระบบดึงและประมวลผลข้อมูลการพยากรณ์จากแบบจำลองสภาพอากาศโลก (NWP Models) ตามรอบการรัน (Forecast Cycles) และข้อมูลภาพถ่ายดาวเทียมอุตุนิยมวิทยา Himawari-9 เพื่อสกัดตัวชี้วัดการก่อตัวและการพาความร้อนของเมฆ (Convective Cloud Evolution) โดยมีกฎเหล็กป้องกัน Data Leakage เชิงเวลาอย่างเคร่งครัด  
> **ข้อกำหนดพื้นที่ทำงาน**: สคริปต์ (`src/data/nwp_downloader.py`, `src/data/himawari_extractor.py`) และแคชข้อมูล (`data/raw/nwp_runs/`, `data/raw/himawari9/`) ต้องอยู่ภายใต้ `weather-forcast-enhance/` ทั้งหมด ไม่มีการเรียกใช้จากภายนอก

---

## 1. การจัดโครงสร้างแบบจำลองสภาพอากาศตาม Forecast Cycle (TIGGE Archive Pipeline)

ตามข้อกำหนดใน [`weather-forcast-enhance/plan.md`](file:///e:/water-analysis-project/weather-forcast-enhance/plan.md) ระบบจะจัดเก็บและประมวลผลข้อมูลการพยากรณ์โดยใช้ **Forecast Run Cycle เป็นแกนหลัก**:

```text
forecast_run_time (รอบเวลาที่โมเดลเริ่มรัน เช่น 00:00, 12:00 UTC)
forecast_valid_time (เวลาเป้าหมายที่ผลพยากรณ์นั้นมีผลบังคับใช้)
lead_time = forecast_valid_time - forecast_run_time (เช่น 6h, 12h, 18h, 24h, ...)
```

### 1.1 การเลือกใช้ TIGGE Archive (ECDS / Copernicus CDS) เป็นแหล่งข้อมูลหลัก

เนื่องจากการตรวจสอบพบว่า Open-Meteo ขาดข้อมูลประวัติย้อนหลังในปี 2021–2022 สำหรับหลายโมเดล โครงการจึงกำหนดให้ใช้ **TIGGE (The International Grand Global Ensemble)** ภายใต้ **ECMWF Data Store (ECDS)** เป็นคลังข้อมูลหลักสำหรับดึง Historical Forecast Runs ย้อนหลัง 2021–2025:

* **จุดเด่น**: มี Historical Forecast Cycles ครบถ้วน **100% ตลอดปี 2021–2025** จากศูนย์พยากรณ์ชั้นนำของโลกทั้ง 6 ค่าย
* **รหัสศูนย์พยากรณ์ใน TIGGE (`origin`) ครอบคลุม 6 โมเดลเป้าหมาย**:

| ลำดับ | โมเดลพยากรณ์ | ประเทศ/สังกัด | TIGGE Origin Code | ขอบเขต & ความพร้อม (2021–2025) |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **ECMWF IFS** | ยุโรป (ECMWF) | `ecmf` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |
| 2 | **NOAA NCEP GFS** | สหรัฐอเมริกา (NOAA) | `kwbc` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |
| 3 | **CMC GEM Global** | แคนาดา (ECCC) | `cwao` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |
| 4 | **BoM ACCESS-G** | ออสเตรเลีย (BoM) | `ammc` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |
| 5 | **DWD ICON Global** | เยอรมนี (DWD) | `edzw` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |
| 6 | **Météo-France ARPEGE** | ฝรั่งเศส (Météo-France) | `lfpw` | ✅ มีครบสมบูรณ์ทุกวันตลอด 2021–2025 |

### 1.2 สเปกการดึงข้อมูลผ่าน API (TIGGE API Specifications via `cdsapi`)

* **Client Library**: `cdsapi` (เวอร์ชัน `>= 0.7.7`) เชื่อมต่อ URL `https://ecds.ecmwf.int/api`
* **Dataset Identifier**: `tigge` (Product type: `control` สำหรับ Deterministic Single Run หรือ `ensemble_mean`)
* **Spatial Bounding Box (ตัดเฉพาะพื้นที่ประเทศไทย)**:
  - `area`: `[21.0, 97.0, 5.0, 106.0]` (North: 21°N, West: 97°E, South: 5°N, East: 106°E)
  - *ประโยชน์*: ลดปริมาณข้อมูลดาวน์โหลดลงมากกว่า 98% เมื่อเทียบกับไฟล์ทั้งโลก ทำให้ดาวน์โหลดได้รวดเร็วและใช้พื้นที่จัดเก็บน้อยมาก
* **รอบเวลาพยากรณ์ (`time`)**: `['00:00', '12:00']` UTC (รอบหลักระดับสากล)
* **Lead-time Steps (`step`)**: `['6', '12', '18', '24', '30', '36', '42', '48']` ชั่วโมง
* **ตัวแปรพยากรณ์ (TIGGE Input Variables)**:
  - `total_precipitation` (ปริมาณฝนสะสม - $tp$ แปลงเป็น mm รายช่วงชั่วโมง)
  - `surface_pressure` (ความกดอากาศผิวพื้น - $sp$ หน่วย Pa $\rightarrow$ hPa)
  - `2m_temperature` (อุณหภูมิ 2 เมตร - $2t$ หน่วย K $\rightarrow$ °C)
  - `10m_u_component_of_wind` ($u_{10}$ ลมแนวระนาบ)
  - `10m_v_component_of_wind` ($v_{10}$ ลมแนวตั้ง)
  - `convective_available_potential_energy` ($cape$ - หากมีใน catalog ของศูนย์นั้นๆ)

### 1.3 กระบวนการประมวลผลไฟล์ (GRIB2 to Analysis-Ready Parquet)

1. สคริปต์ `src/data/tigge_downloader.py` ทำการดาวน์โหลดไฟล์ GRIB2 เฉพาะ Bounding Box ประเทศไทย
2. สคริปต์ `src/data/tigge_extractor.py` ใช้ `xarray` + `cfgrib` เปิดอ่านไฟล์ แปลงหน่วย และ Interpolate/Bilinear Mapping เข้าสู่โครงข่าย **Thailand 2 km × 2 km Master Grid**
3. บันทึกผลลัพธ์เป็นตาราง Parquet แยกตามโมเดล:
   `weather-forcast-enhance/data/nwp_forecasts/{origin}/run_{YYYYMMDD}_{HHz}.parquet`

---

## 2. การประมวลผลข้อมูลดาวเทียม Himawari-9 (Satellite Cloud Features)

ภาพถ่ายดาวเทียม Geostationary Himawari-9 (JMA) ครอบคลุมประเทศไทยด้วยความถี่ทุก 10 นาที (Full Disk / Target Area) ความละเอียดเชิงพื้นที่ระดับ 2 km:

### 2.1 แบนด์สเปกตรัมที่เลือกใช้ (Target Spectral Bands)
1. **Band 13 (Clean IR Window ~ 10.4 µm)**:
   - วัดอุณหภูมิยอดเมฆ (Cloud Top Brightness Temperature - $BT$)
   - ค่ายิ่งต่ำ (เช่น $BT < 210\text{ K}$ หรือ $-63^\circ\text{C}$) แสดงถึงยอดเมฆที่สูงมาก เมฆก่อตัวในแนวตั้งรุนแรง (Deep Convective Cloud / Cumulonimbus) ซึ่งสัมพันธ์กับฝนตกหนัก
2. **Band 8 (Upper-level Water Vapor ~ 6.2 µm)**:
   - วัดไอน้ำและความชื้นในชั้นโทรโพสเฟียร์ตอนบน ชี้วัดการยกตัวของมวลอากาศ (Vertical Updrafts)

### 2.2 การสกัดฟีเจอร์การวิวัฒนาการของเมฆ (Cloud Evolution Features)
สำหรับแต่ละจุดศูนย์กลาง 2 km Grid ในรัศมีโดยรอบ:
- **Spatial Aggregation ณ เวลา $t$**:
  - $BT_{\text{mean}}$, $BT_{\text{std}}$, $BT_{\text{min}}$, $BT_{\text{max}}$
- **Temporal Lag Features (การเปลี่ยนแปลงย้อนหลัง)**:
  - $BT(t)$ (เวลาปัจจุบันก่อนปล่อยพยากรณ์)
  - $BT(t - 10\text{m})$ (ย้อนหลัง 10 นาที)
  - $BT(t - 20\text{m})$ (ย้อนหลัง 20 นาที)
  - $BT(t - 30\text{m})$ (ย้อนหลัง 30 นาที)
- **อัตราการลดลงของอุณหภูมิยอดเมฆ (Cloud Cooling Rate / Convective Trigger)**:
  $$\Delta BT_{30} = BT(t) - BT(t - 30\text{m})$$
  *(หาก $\Delta BT$ ติดลบมาก แสดงว่ายอดเมฆกำลังพุ่งสูงขึ้นอย่างรวดเร็ว โอกาสเกิดฝนฟ้าคะนองฉับพลันสูง)*

---

## 3. กฎเหล็กป้องกัน Data Leakage เชิงเวลา (Temporal Anti-Leakage Protocol)

> [!CAUTION]
> **กฎการตัดเวลา (Look-ahead Prevention Invariant):**  
> ณ จุดเวลา `forecast_run_time` ใดๆ แบบจำลอง ML Bias Correction **มีสิทธิ์เข้าถึงเฉพาะข้อมูลสังเกตการณ์ (ทั้ง Ground และ Satellite) ที่เกิดขึ้นก่อนหรือตรงกับ `run_time` เท่านั้น!**

```text
ตัวอย่างการทดสอบ:
  Forecast Run Time : 2025-06-01 00:00 UTC
  Forecast Valid Time: 2025-06-01 06:00 UTC (Lead time = +6 ชั่วโมง)

  ✅ อนุญาตให้ใช้:
     - NWP Forecast สำหรับ valid 06:00 (เพราะโมเดล NWP ปล่อยออกมาตั้งแต่ run 00:00)
     - Himawari-9 BT ณ เวลา 00:00, 23:50, 23:40, 23:30 (บันทึกก่อน run time)
     - Ground Observation ณ เวลา 00:00 (บันทึกก่อน run time)

  ❌ ห้ามใช้เด็ดขาด (ถือเป็น Leakage):
     - Himawari-9 ณ เวลา 01:00, 02:00, ..., 06:00 (เพราะในสถานการณ์พยากรณ์จริง เวลายังไม่เกิดขึ้น)
     - Ground Observation ณ เวลา 01:00, ..., 06:00
```

การควบคุมใน Pipeline: ทุกฟีเจอร์ของ Himawari และ Ground จะต้องมี metadata stamp กำกับชัดเจนว่า `observation_time <= run_time` เสมอ

---

## 4. โครงสร้างการจัดเก็บข้อมูลและการแปลงหน่วย (Data Storage & Caching)

```text
weather-forcast-enhance/
└── data/
    ├── nwp_forecasts/
    │   ├── ecmf/               # ECMWF IFS (TIGGE Origin)
    │   │   └── run_20210101_00z.parquet
    │   ├── kwbc/               # NOAA NCEP GFS (TIGGE Origin)
    │   │   └── run_20210101_00z.parquet
    │   ├── cwao/               # CMC GEM Global (TIGGE Origin)
    │   │   └── run_20210101_00z.parquet
    │   ├── ammc/               # BoM ACCESS-G (TIGGE Origin)
    │   │   └── run_20210101_00z.parquet
    │   ├── edzw/               # DWD ICON Global (TIGGE Origin)
    │   │   └── run_20210101_00z.parquet
    │   └── lfpw/               # Météo-France ARPEGE (TIGGE Origin)
    │       └── run_20210101_00z.parquet
    └── himawari9/
        ├── band13_bt/
        │   └── 2021/01/01/h9_b13_20210101_0000.parquet
        └── band08_wv/
            └── 2021/01/01/h9_b08_20210101_0000.parquet
```

---

## 5. สรุป Checklist ความพร้อม Phase 2

- [ ] ตั้งค่าการเชื่อมต่อ `cdsapi` สำหรับ ECDS (`https://ecds.ecmwf.int/api`) และระบุ Origin ทั้ง 6 ค่าย
- [ ] กำหนด Bounding Box ตัดเฉพาะประเทศไทย `[21.0, 97.0, 5.0, 106.0]` เพื่อความรวดเร็วและประหยัดพื้นที่
- [ ] ออกแบบโมดูลแปลง GRIB2 เป็น Dataframe/Parquet บน 2 km Grid (`src/data/tigge_extractor.py`)
- [ ] กำหนดสูตรการคำนวณ `lead_time = valid_time - run_time`
- [ ] ออกแบบโมดูลสกัดแบนด์ 13 และ 8 ของ Himawari-9 และการคำนวณ $\Delta BT$
- [ ] ออกแบบ Unit Test ตรวจสอบเงื่อนไข Anti-Leakage (ทดสอบว่าไม่มี Timestamp หลัง `run_time` เล็ดลอดเข้าไปในโมเดล)
