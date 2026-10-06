# 📑 Data & Feature Dictionary: Training Feature Specification

> **เอกสารอ้างอิง**: [`weather-forcast-enhance/plan.md`](file:///e:/water-analysis-project/weather-forcast-enhance/plan.md)  
> **ที่จัดเก็บข้อมูล Clean Parquet**: `weather-forcast-enhance/data/clean_parquet/`  
> **ที่จัดเก็บ Feature Matrix**: `weather-forcast-enhance/data/features/`  
> **ขอบเขตพื้นที่ทำงาน**: อยู่ภายใต้ `weather-forcast-enhance/` ทั้งหมด 100%

เอกสารฉบับนี้ระบุ **รายการฟีเจอร์ทั้งหมด (Complete Feature Catalog)** ที่ใช้ในการฝึกสอนแบบจำลอง Machine Learning สำหรับงาน Precipitation Bias Correction บน 2 km × 2 km Grid

---

## 1. ตารางสรุปภาพรวมฟีเจอร์แยกตามกลุ่ม (Feature Groups Summary)

```text
┌─────────────────────────────────┬───────────────────────────────────────────────────────────┐
│ กลุ่มฟีเจอร์ (Feature Group)    │ รายละเอียดและแหล่งที่มา (Source & Description)             │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┤
│ Group 1: Temporal & Cycle       │ รอบพยากรณ์, Valid Time, Lead Time, Diurnal & Seasonal     │
│ Group 2: NWP Weather Model      │ ผลพยากรณ์จากแบบจำลองสภาพอากาศโลก (ECMWF IFS/AIFS, NCEP)   │
│ Group 3: Ground Spatial Obs     │ สังเกตการณ์ภาคพื้นดิน HII (ฝน, ความกด, ความชื้น, ระยะห่าง)   │
│ Group 4: Himawari-9 Satellite   │ อุณหภูมิยอดเมฆ IR BT Band 13, ไอน้ำ Band 8, ΔBT 30 นาที    │
│ Group 5: Topographic & Geo      │ ภูมิประเทศ ความสูง DEM, ความลาดชัน Slope, ทิศทาง Aspect    │
└─────────────────────────────────┴───────────────────────────────────────────────────────────┘
```

---

## 2. รายละเอียดฟีเจอร์ทั้งหมดแบบเจาะลึก (Feature Specification Table)

### 🕒 Group 1: Temporal & Lead-Time Metadata (ตัวแปรเวลาและรอบพยากรณ์)

| ชื่อฟีเจอร์ (Feature Name) | ชนิดข้อมูล | หน่วย | คำอธิบายและความสำคัญทางอุตุนิยมวิทยา | กฎการตัดเวลา (Anti-Leakage) |
|---|---|:---:|---|:---:|
| `run_time` | timestamp | UTC | รอบเวลาที่แบบจำลอง NWP เริ่มรัน (เช่น 00Z, 06Z, 12Z, 18Z) | Identifier |
| `valid_time` | timestamp | UTC | เวลาเป้าหมายที่ผลพยากรณ์มีผลบังคับใช้ | Identifier |
| `lead_time` | integer | hours | ระยะเวลานำหน้า: $\text{lead\_time} = \text{valid\_time} - \text{run\_time}$ (1h ถึง 24h) | รู้ได้ล่วงหน้า |
| `hour_of_day_sin` | float32 | - | $\sin\left(\frac{2\pi \times \text{hour}}{24}\right)$ วัฏจักรความร้อนประจำวัน (Diurnal Cycle) | รู้ได้ล่วงหน้า |
| `hour_of_day_cos` | float32 | - | $\cos\left(\frac{2\pi \times \text{hour}}{24}\right)$ มักเกิดฝนฟ้าคะนองช่วงบ่าย-ค่ำ | รู้ได้ล่วงหน้า |
| `day_of_year_sin` | float32 | - | $\sin\left(\frac{2\pi \times \text{day}}{365.25}\right)$ วัฏจักรฤดูกาล (Seasonal Monsoon Cycle) | รู้ได้ล่วงหน้า |
| `day_of_year_cos` | float32 | - | $\cos\left(\frac{2\pi \times \text{day}}{365.25}\right)$ แยกฤดูฝน (มรสุมตะวันตกเฉียงใต้) กับฤดูแล้ง | รู้ได้ล่วงหน้า |

---

### 🌐 Group 2: Weather Model Forecast Features (แบบจำลองสภาพอากาศโลก - TIGGE Archive)
*ดึงมาจาก TIGGE 6 โมเดลหลัก: ECMWF (`ecmf`), NCEP GFS (`kwbc`), CMC GEM (`cwao`), BoM ACCESS-G (`ammc`), DWD ICON (`edzw`), Météo-France ARPEGE (`lfpw`) ที่ Valid ณ เวลา `valid_time`*

| ชื่อฟีเจอร์ (Feature Name) | ชนิดข้อมูล | หน่วย | คำอธิบาย |
|---|---|:---:|---|
| `nwp_precip_raw` | float32 | mm/h | ปริมาณฝนพยากรณ์รายชั่วโมงจากแบบจำลอง (Raw Forecast) |
| `nwp_temp_2m` | float32 | °C | อุณหภูมิอากาศที่ระดับ 2 เมตร |
| `nwp_rh_2m` | float32 | % | ความชื้นสัมพัทธ์ที่ระดับ 2 เมตร |
| `nwp_surface_pressure` | float32 | hPa | ความกดอากาศที่ระดับผิวพื้น |
| `nwp_u10` | float32 | m/s | เวกเตอร์ลมแกนแนวระนาบ ตะวันออก-ตะวันตก ที่ระดับ 10 เมตร |
| `nwp_v10` | float32 | m/s | เวกเตอร์ลมแกนแนวตั้ง เหนือ-ใต้ ที่ระดับ 10 เมตร |
| `nwp_wind_speed_10m` | float32 | m/s | ความเร็วลมรวม $\sqrt{u_{10}^2 + v_{10}^2}$ |
| `nwp_total_cloud_cover` | float32 | 0–1 | สัดส่วนเมฆปกคลุมท้องฟ้ารวม |
| `nwp_cape` | float32 | J/kg | Convective Available Potential Energy (พลังงานศักย์การพาความร้อน) |

---

### 📡 Group 3: Spatial Ground Observation Features (สถานีภาคพื้นดิน HII)
*คำนวณจากสถานีตรวจวัด HII ที่ผ่านการตรวจสอบ Zero-Null จาก Phase 1*  
> [!IMPORTANT]
> **กฎ Anti-Leakage และ Spatial Barrier**:  
> 1. ต้องใช้เฉพาะค่าสังเกตการณ์ที่บันทึก ณ เวลา $t \le \text{run\_time}$ เท่านั้น  
> 2. สถานีที่อยู่ห่างจาก Target Grid ใจกลาง $< 2\text{ km}$ **จะไม่ถูกนำมาใช้** เพื่อป้องกันโมเดลท่องจำคำตอบ  
> 3. ข้อมูลสังเกตการณ์ภาคพื้นดินสกัดจากเครือข่ายสถานี **HII เท่านั้น** โดย **ห้ามนำข้อมูลสถานี DWR ใน `dataset/dwr_rain/` มาใช้สร้างฟีเจอร์เด็ดขาด** (สงวน DWR ไว้เป็น 100% Blind Spatial Hold-Out ใน Phase 5 เท่านั้น)

| ชื่อฟีเจอร์ (Feature Name) | ชนิดข้อมูล | หน่วย | คำอธิบายและรายละเอียดการคำนวณ |
|---|---|:---:|---|
| `rain_mean_2_5km` | float32 | mm/h | ปริมาณฝนเฉลี่ยของสถานีในรัศมีวง 2–5 km ณ เวลา $t \le \text{run\_time}$ |
| `rain_max_2_5km` | float32 | mm/h | ปริมาณฝนสูงสุดในรัศมีวง 2–5 km |
| `pressure_mean_2_5km` | float32 | hPa | ความกดอากาศเฉลี่ยในรัศมีวง 2–5 km |
| `pressure_std_2_5km` | float32 | hPa | ความแปรปรวนของความกดอากาศในรัศมี 2–5 km |
| `humidity_mean_2_5km` | float32 | % | ความชื้นสัมพัทธ์เฉลี่ยในรัศมีวง 2–5 km |
| `rain_mean_5_10km` | float32 | mm/h | ปริมาณฝนเฉลี่ยของสถานีในรัศมีวง 5–10 km |
| `pressure_mean_5_10km` | float32 | hPa | ความกดอากาศเฉลี่ยในรัศมีวง 5–10 km |
| `humidity_mean_5_10km` | float32 | % | ความชื้นสัมพัทธ์เฉลี่ยในรัศมีวง 5–10 km |
| `pressure_mean_10_20km` | float32 | hPa | ความกดอากาศเฉลี่ยระดับกว้าง 10–20 km |
| `humidity_mean_10_20km` | float32 | % | ความชื้นสัมพัทธ์เฉลี่ยระดับกว้าง 10–20 km |
| `pressure_mean_20_50km` | float32 | hPa | ความกดอากาศเฉลี่ยระดับภูมิภาค 20–50 km |
| `nearest_stn_dist_km` | float32 | km | ระยะทางไปยังสถานีตรวจวัดที่ใกล้ที่สุด (ซึ่งต้อง $\ge 2\text{ km}$) |
| `nearest_stn_bearing_sin` | float32 | - | $\sin(\theta)$ มุมทิศทางไปยังสถานีตรวจวัดที่ใกล้ที่สุด |
| `nearest_stn_bearing_cos` | float32 | - | $\cos(\theta)$ มุมทิศทางไปยังสถานีตรวจวัดที่ใกล้ที่สุด |
| `pressure_gradient_mag` | float32 | hPa/km | $\|\nabla P\|$ เกรเดียนต์ความกดอากาศเชิงพื้นที่ |
| `humidity_gradient_mag` | float32 | %/km | $\|\nabla RH\|$ เกรเดียนต์ความชื้นสัมพัทธ์เชิงพื้นที่ |

---

### 🛰️ Group 4: Himawari-9 Satellite Cloud Features (ภาพถ่ายดาวเทียมอุตุนิยมวิทยา)
*สกัดจากภาพถ่ายดาวเทียม Geostationary Himawari-9 เหนือกริดเป้าหมาย*  
> [!IMPORTANT]
> **กฎ Anti-Leakage**: ต้องใช้เฉพาะภาพที่ถ่าย ณ เวลา $t \le \text{run\_time}$ เท่านั้น

| ชื่อฟีเจอร์ (Feature Name) | ชนิดข้อมูล | หน่วย | คำอธิบายและรายละเอียดทางฟิสิกส์ |
|---|---|:---:|---|
| `bt_b13_mean` | float32 | K | อุณหภูมิยอดเมฆเฉลี่ย Band 13 (Clean IR Window ~10.4 µm) ณ เวลา $t$ |
| `bt_b13_min` | float32 | K | อุณหภูมิยอดเมฆต่ำสุดในกริด (ค่ายิ่งต่ำ ยอดเมฆยิ่งสูงมาก เสี่ยงฝนตกหนัก) |
| `bt_b13_std` | float32 | K | ความแปรปรวนของยอดเมฆ (ชี้วัดขอบเขตของแนวเมฆฝน) |
| `bt_b13_lag10` | float32 | K | อุณหภูมิยอดเมฆย้อนหลัง 10 นาที: $BT(t - 10\text{m})$ |
| `bt_b13_lag20` | float32 | K | อุณหภูมิยอดเมฆย้อนหลัง 20 นาที: $BT(t - 20\text{m})$ |
| `bt_b13_lag30` | float32 | K | อุณหภูมิยอดเมฆย้อนหลัง 30 นาที: $BT(t - 30\text{m})$ |
| `bt_b13_delta30` | float32 | K | **อัตราการลดลงของอุณหภูมิยอดเมฆ**: $\Delta BT = BT(t) - BT(t - 30\text{m})$ (หากติดลบมาก = Convective Updraft รุนแรง) |
| `bt_b08_mean` | float32 | K | อุณหภูมิไอน้ำ Band 8 (Upper-level Water Vapor ~6.2 µm) ชี้วัดความชื้นชั้นบน |

---

### ⛰️ Group 5: Topographic & Geographic Features (ภูมิประเทศและกายภาพ)
*คงที่ตามตำแหน่งของแต่ละ 2 km Grid Cell*

> [!CAUTION]
> **ข้อควรระวัง: ปัญหาการจำพิกัด (Spatial Coordinate Memorization / Overfitting Risk):**  
> หากป้อน `target_lat` และ `target_lon` ดิบเข้าไปใน Decision Trees (เช่น LightGBM, CatBoost) ต้นไม้ตัดสินใจจะสร้างเงื่อนไขแบบตัดแกนตรง (`lat > 13.5 AND lon < 100.2`) ซึ่งจะทำให้**โมเดลสร้าง Bounding Box ท่องจำตำแหน่งของสถานีตรวจวัดเฉพาะจุด** แทนที่จะเรียนรู้กฎฟิสิกส์บรรยากาศ เมื่อนำไปพยากรณ์กริด 2 km ที่ไม่มีสถานี จะเกิดรอยต่อขั้นบันได (Spatial Discontinuity Artifacts)  
> **แนวทางแก้ไขตามหลักการ (Best Practice Solution)**:  
> 1. `target_lat, target_lon` จะถูกเก็บเป็น **Spatial Keys / Metadata สำหรับทำ Join และ Mapping เท่านั้น** (ไม่ส่งเข้า Tree Splitting)  
> 2. แทนที่ด้วย **ตัวแปรทางฟิสิกส์และภูมิประเทศที่ต่อเนื่อง (Continuous Physical Proxies)**: เช่น ความสูง (Elevation), ความลาดชัน (Slope), ระยะห่างทะเล (Distance to Coast), และแรงโคริออลิส (Coriolis Parameter)  
> 3. หากต้องการให้โมเดลรับรู้ทิศทางลมมรสุมหรือ Macro-Zone ระดับประเทศ ให้ใช้ **Low-Frequency Spatial Fourier Embeddings (ความถี่ต่ำมาก ระดับ > 200 km)** เพื่อไม่ให้โมเดลสามารถเจาะจงระดับเซลล์ 2 km ได้

| ชื่อฟีเจอร์ (Feature Name) | ชนิดข้อมูล | หน่วย | คำอธิบายและบทบาททางฟิสิกส์ | สถานะการนำไปเทรน |
|---|---|:---:|---|:---:|
| `target_lat` | float32 | deg | ละติจูดของจุดศูนย์กลางกริด ($5.5^\circ\text{N} - 20.5^\circ\text{N}$) | **Metadata Only** (ไม่เข้าโมเดล) |
| `target_lon` | float32 | deg | ลองจิจูดของจุดศูนย์กลางกริด ($97.3^\circ\text{E} - 105.7^\circ\text{E}$) | **Metadata Only** (ไม่เข้าโมเดล) |
| `coriolis_param` | float32 | $s^{-1}$ | แรงโคริออลิส $f = 2\Omega \sin(\text{lat})$ (แปรตามละติจูดเชิงฟิสิกส์ต่อเนื่อง) | ✅ ใช้เทรน |
| `elevation_mean` | float32 | m | ความสูงเฉลี่ยเหนือระดับน้ำทะเลจาก SRTM DEM (ชี้วัด Orographic Lifting) | ✅ ใช้เทรน |
| `elevation_std` | float32 | m | ความขรุขระของภูมิประเทศ (Topographic Roughness) | ✅ ใช้เทรน |
| `slope_deg` | float32 | deg | ความลาดชันของพื้นที่ (องศา) | ✅ ใช้เทรน |
| `aspect_sin` | float32 | - | $\sin(\text{aspect})$ ทิศทางความลาดชัน (ด้านรับลมมรสุม vs ด้านอับลม) | ✅ ใช้เทรน |
| `aspect_cos` | float32 | - | $\cos(\text{aspect})$ | ✅ ใช้เทรน |
| `dist_coast_km` | float32 | km | ระยะห่างจากแนวชายฝั่งทะเลที่ใกล้ที่สุด (ชี้วัดความเป็นแผ่นดิน/ทะเล) | ✅ ใช้เทรน |
| `macro_spatial_fourier_sin` | float32 | - | $\sin\left(\frac{2\pi \times \text{lat}}{L_{\text{macro}}}\right)$ คลื่นความถี่ต่ำ ($L \ge 200\text{ km}$) คุมโซนภูมิอากาศ | ✅ (ตัวเลือกเสริม) |

---

## 3. เป้าหมายการเรียนรู้และ Residual Target (Learning Targets)

| ชื่อคอลัมน์ (Column Name) | ชนิดข้อมูล | หน่วย | คำอธิบาย |
|---|---|:---:|---|
| `observed_rain` | float32 | mm/h | ปริมาณฝนจริงที่วัดได้จากสถานีภาคพื้นดิน ณ `valid_time` (Ground Truth) |
| `target_bias` | float32 | mm/h | **ค่า Residual Bias ที่โมเดลต้องเรียนรู้**: $\text{target\_bias} = \text{observed\_rain} - \text{nwp\_precip\_raw}$ |
| `target_rain_class` | integer | {0, 1} | ป้ายกำกับสำหรับ Stage 1 ของ Hurdle Model: $\mathbb{I}(\text{observed\_rain} \ge 0.1\text{ mm})$ |

---

## 4. แผนผังการใช้ฟีเจอร์ในการทดลองแต่ละรอบ (Feature Allocation across Ablation Variants)

ตารางด้านล่างแสดงว่าการทดลองในแต่ละ Variant ($M_1, M_2, M_3, M_4$) ใช้ฟีเจอร์กลุ่มใดบ้าง:

| กลุ่มฟีเจอร์ (Feature Group) | Baseline ($B_0$) | $M_1$ (Weather + Ground) | $M_2$ (Weather + Satellite) | $M_3$ (Weather + Ground + Satellite) | $M_4$ (Full Spatial Architecture) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Group 1: Temporal & Lead-Time** | ❌ *(Raw)* | ✅ | ✅ | ✅ | ✅ |
| **Group 2: NWP Forecasts** | ✅ *(Precip Only)* | ✅ *(All NWP)* | ✅ *(All NWP)* | ✅ *(All NWP)* | ✅ *(All NWP)* |
| **Group 3: Ground Spatial Obs** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Group 4: Himawari-9 Satellite** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Group 5: Topographic & Geo** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Spatial Gradients ($\nabla P, \nabla RH$)** | ❌ | ❌ | ❌ | ❌ | ✅ *(เพิ่มเฉพาะ M4)* |

---

## 5. สรุปข้อสรุปสำคัญทางวิศวกรรมข้อมูล (Data Invariants)

1. **จำนวนฟีเจอร์รวมสำหรับการฝึกสอน**:
   - $M_1$ Variant: ~25 ฟีเจอร์
   - $M_2$ Variant: ~22 ฟีเจอร์
   - $M_3$ Variant: ~33 ฟีเจอร์
   - $M_4$ Variant: ~36 ฟีเจอร์ (Full Feature Vector)
2. **การจัดเก็บข้อมูล (File Storage Layout)**:
   - ตารางฟีเจอร์ทั้งหมดจะถูกสร้างและจัดเก็บในรูปแบบ Apache Parquet (Snappy Compressed) ภายใต้:
     `weather-forcast-enhance/data/features/train_2021_2024.parquet`
     `weather-forcast-enhance/data/features/test_2025.parquet`
   - ไม่มีคอลัมน์ใดที่อ้างอิงหรือพึ่งพาไฟล์นอก `weather-forcast-enhance/`
