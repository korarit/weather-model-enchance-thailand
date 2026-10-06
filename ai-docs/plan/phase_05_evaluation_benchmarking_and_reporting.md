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

### 2.1 ดัชนีวัดความต่อเนื่องเชิงปริมาณ (Continuous Quantitative Metrics: RMSE & MAE Focus)

การประเมินความคลาดเคลื่อนต่อเนื่องของปริมาณน้ำฝนจะใช้ **RMSE และ MAE เป็นแกนหลักควบคู่กัน** เนื่องจากสะท้อนพฤติกรรมความผิดพลาดคนละมิติทางอุตุนิยมวิทยา:

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (R_{\text{pred}, i} - R_{\text{obs}, i})^2} \quad (\text{mm/h})$$
   - **ความหมายทางกายภาพ**: ลงโทษความผิดพลาดขนาดใหญ่แบบยกกำลังสอง ($\text{squared penalization}$) จึงมีความไวสูงต่อเหตุการณ์ฝนตกหนักหลุดเป้า (Extreme Outliers) เหมาะสำหรับชี้วัดว่าโมเดลใดสามารถควบคุมความคลาดเคลื่อนรุนแรงในพายุฝนได้ดีที่สุด
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |R_{\text{pred}, i} - R_{\text{obs}, i}| \quad (\text{mm/h})$$
   - **ความหมายทางกายภาพ**: ลงโทษความผิดพลาดเชิงเส้นตรง ($\text{linear penalization}$) ไม่ไวต่อค่ากระโดด จึงสะท้อนระดับความคลาดเคลื่อนเฉลี่ยตามจริงในสภาวะปกติ (Typical Expected Error) ได้อย่างเป็นกลาง
3. **อัตราส่วนความแปรปรวนของความคลาดเคลื่อน (RMSE-to-MAE Ratio)**:
   $$\text{Ratio} = \frac{\text{RMSE}}{\text{MAE}}$$
   - หาก $\text{Ratio} \to 1.0$: ความคลาดเคลื่อนสม่ำเสมอ ไม่มีค่าสุดโต่ง (Uniform Error)
   - หาก $\text{Ratio} \gg 1.25$: บ่งชี้ว่ามีความแปรปรวนของความคลาดเคลื่อนสูงมาก มีเหตุการณ์ที่โมเดลพยากรณ์พลาดรุนแรงเป็นครั้งคราว (Presence of Severe Outliers)
4. **ดัชนีวัดการพัฒนาทักษะ (Skill Score: % Error Reduction เทียบกับ Raw NWP Baseline $B_0$)**:
   $$\text{SS}_{\text{RMSE}} = \left(1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{raw}}}\right) \times 100\%$$
   $$\text{SS}_{\text{MAE}} = \left(1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{raw}}}\right) \times 100\%$$
   - ค่าเป็นบวก ($\text{SS} > 0\%$): การทำ Bias Correction สามารถลด Error ได้ดีกว่าโมเดลสภาพอากาศโลกเดิม
   - ค่าเป็นลบ ($\text{SS} < 0\%$): โมเดลทำให้เกิดความคลาดเคลื่อนแย่ลงกว่าเดิม (Degradation)
5. **Mean Bias Error (MBE / Systematic Bias)**:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (R_{\text{pred}, i} - R_{\text{obs}, i}) \quad (\text{mm/h})$$
   *(ค่าบวก = พยากรณ์เกินจริง / Overestimation, ค่าลบ = พยากรณ์ต่ำกว่าจริง / Underestimation)*
6. **Pearson Correlation ($r$) & Spearman Rank Correlation ($\rho$)**:
   - วัดความสอดคล้องเชิงแนวโน้มและการจัดลำดับของปริมาณฝน

---

### 2.2 กรอบการเปรียบเทียบเชิงลึกของ RMSE และ MAE (Comprehensive RMSE & MAE Comparison Framework)

ระบบจะทำการเปรียบเทียบ RMSE และ MAE อย่างละเอียดครอบคลุม **5 มิติหลัก**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    มิติการเปรียบเทียบเชิงลึกของ RMSE และ MAE (5-Dimensional Matrix)        │
├───────────────────────────────┬─────────────────────────────────────────────────────────┤
│ มิติที่ 1: Model vs Raw NWP   │ เปรียบเทียบ % Error Reduction (SS_RMSE, SS_MAE) ของทั้ง   │
│    (Benchmark vs Baseline B0) │ 18 การทดลอง (6 Models x 3 Variants) เทียบกับ Raw Model   │
├───────────────────────────────┼─────────────────────────────────────────────────────────┤
│ มิติที่ 2: Cross-Architecture │ เปรียบเทียบความแม่นยำระหว่าง 6 สถาปัตยกรรม:               │
│    (เปรียบเทียบโครงสร้างโมเดล)│ LightGBM vs CatBoost vs Hurdle GBDT vs Multi-Quantile   │
│                               │ vs ST-GNN vs 2D U-Net เพื่อหาสถาปัตยกรรมที่ Error ต่ำสุด  │
├───────────────────────────────┼─────────────────────────────────────────────────────────┤
│ มิติที่ 3: Lead-Time Horizon  │ วิเคราะห์การเพิ่มขึ้นของ RMSE และ MAE ตามระยะเวลาล่วงหน้า  │
│    (เสื่อมถอยตามระยะเวลานำหน้า)│ ตั้งแต่ +1h จนถึง +24h หาจุดตัดที่ ML ยังชนะ Raw NWP     │
├───────────────────────────────┼─────────────────────────────────────────────────────────┤
│ มิติที่ 4: Basin Heterogeneity│ วิเคราะห์ RMSE และ MAE แยกตาม 25 ลุ่มน้ำหลัก              │
│    (เปรียบเทียบแยกเชิงพื้นที่)   │ เทียบพื้นที่ภูเขาสูง (ยม, น่าน, ปิง) กับที่ราบลุ่ม/ชายฝั่ง│
├───────────────────────────────┼─────────────────────────────────────────────────────────┤
│ มิติที่ 5: Generalization Gap │ เปรียบเทียบ RMSE และ MAE บนชุดสถานี HII (In-Network)     │
│    (วัดความสามารถข้ามเครือข่าย) │ กับสถานี DWR (100% Blind Spatial Hold-Out):             │
│                               │ Δ_Gen = Error(DWR) - Error(HII) พิสูจน์ Zero-Leakage   │
└───────────────────────────────┴─────────────────────────────────────────────────────────┘
```

---

### 2.3 ดัชนีวัดการตรวจจับฝนตามระดับความรุนแรง (Rainfall Threshold & Contingency Metrics)
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

## 3. การวิเคราะห์ตาม Lead Time (Lead-Time Horizon Degradation: RMSE & MAE)

ตามข้อกำหนดใน [`weather-forcast-enhance/plan.md`](file:///e:/water-analysis-project/weather-forcast-enhance/plan.md) ข้อ 14:

ผลการพยากรณ์จะถูกแจกแจงตามค่า `lead_time`:
$$\text{lead\_time} \in \{1\text{h}, 2\text{h}, 3\text{h}, \dots, 24\text{h}\}$$

เพื่อเปรียบเทียบพฤติกรรมการเสื่อมถอยของความแม่นยำ (Error Growth Dynamics) ของทั้ง **RMSE** และ **MAE**:

```text
Error (mm/h)
 │
 │      Raw Weather Model RMSE (ECMWF IFS) ───- - - - - -
 │     /
 │    /   Bias-Corrected Model RMSE (M3) ────────────────
 │   /
 │  │    Raw Weather Model MAE  ───- - - - - - - - - - - -
 │  │   /
 │  │  /  Bias-Corrected Model MAE (M3)  ────────────────
 └──┴──────┴──────┴──────┴──────┴──────┴──────┴─────> Lead Time (Hours)
   +1h    +3h    +6h    +9h   +12h   +18h   +24h
```

### คำถามและการวิเคราะห์เปรียบเทียบเชิงลึก:
1. **Short-range Lead Time (+1h ถึง +3h)**:
   - ข้อมูลสถานีภาคพื้นดิน (Ground Obs) และภาพถ่ายดาวเทียม Himawari-9 ช่วยลดทั้ง RMSE และ MAE ลงได้กี่เปอร์เซ็นต์?
   - โมเดลใดสามารถกดอัตราส่วน $\frac{\text{RMSE}}{\text{MAE}} \to 1.0$ ได้ดีที่สุดในช่วงเริ่มต้น?
2. **Medium-range Lead Time (+6h ถึง +12h)**:
   - อิทธิพลของ Ground และ Satellite ค่อยๆ จางลงอย่างไร? จุดตัด (Crossing Point) ที่ Skill Score เริ่มลดลงอยู่ที่ Lead Time เท่าใด?
3. **Longer-range Lead Time (+12h ถึง +24h)**:
   - ที่ระยะนำหน้ายาวขึ้น ML ยังคงรักษาระดับ RMSE และ MAE ให้ต่ำกว่า Raw Weather Model ($B_0$) ได้หรือไม่?
   - สถาปัตยกรรมแบบใด (เช่น U-Net หรือ Hurdle GBDT) ที่รักษาทักษะการลด Error ไว้ได้ยาวนานที่สุด?

---

## 4. การวิเคราะห์แนวโน้มรายเดือนแบบกราฟเส้น (Monthly Seasonal Dynamics: 12-Month Line Charts Suite)

สภาพภูมิอากาศและอุทกวิทยาของประเทศไทยมีความผันแปรเชิงฤดูกาลอย่างรุนแรง (Seasonal Climate Heterogeneity) การประเมินผลค่าเฉลี่ยทั้งปีเพียงอย่างเดียวอาจบดบังจุดแข็งหรือจุดอ่อนสำคัญของแบบจำลอง ดังนั้น **ทุกการเปรียบเทียบในระบบ (All Comparisons) จะต้องมีชุดกราฟเส้นรายเดือน (12-Month Time-Series Line Charts: Jan 2025 – Dec 2025)** ครอบคลุมทั้ง 3 ฤดูกาลหลักของไทย:

1. **ฤดูแล้งและฤดูหนาว (Dry/Cool Season: พ.ย. – ก.พ.)**: โอกาสเกิดฝนต่ำมาก (>95% No-Rain Hours) เป็นช่วงทดสอบว่าโมเดลเกิดอาการเตือนหลอก (False Alarms) หรือพยากรณ์ฝนพร่ำเพรื่อหรือไม่
2. **ฤดูร้อนและช่วงก่อนมรสุม (Pre-Monsoon Season: มี.ค. – เม.ย.)**: มีพายุฤดูร้อน (Convective Thunderstorms) ฝนตกหนักฉับพลันเฉพาะจุด เป็นช่วงทดสอบว่าโมเดลจับ Severe Outliers และภาพถ่ายดาวเทียม Himawari-9 ช่วยลด Error ได้จริงหรือไม่
3. **ฤดูมรสุมและร่องความกดอากาศต่ำ (Monsoon/Wet Season: พ.ค. – ต.ค.)**: ฝนตกชุกครอบคลุมบริเวณกว้าง ปริมาณฝนสะสมสูง เสี่ยงน้ำท่วมฉับพลันและน้ำป่าไหลหลากสูงสุดในรอบปี เป็นช่วงทดสอบความแม่นยำในการเตือนภัยวิกฤติ (Heavy Rain POD) และอัตราการไม่เตือน (Miss Rate)

### 4.1 แผนภาพแนวคิดชุดกราฟเส้นรายเดือน (Monthly Line Chart Architecture)

ทุกกราฟเส้นรายเดือนจะมีแกน X เป็น 12 เดือน (`Jan 2025` ถึง `Dec 2025`) โดยลากเส้นเชื่อมโยงผลการวัดในแต่ละเดือนอย่างต่อเนื่อง:

```text
Metric Value
 │
 │      Raw Weather Model Baseline (B0) ───- - - - - - - - - - - - - - - - -
 │     /
 │    /─── Bias-Corrected Model M1 (Ground Only) ───────────────────────────
 │   /
 │  /───── Bias-Corrected Model M3 (Ground + Himawari-9) ───────────────────
 └──┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴───> Month (2025)
   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
   [──── หน้าแล้ง/หนาว ────]  [ พายุฤดูร้อน ] [───────── ฤดูมรสุม/น้ำหลาก ─────────] [ หน้าแล้ง ]
```

### 4.2 รายละเอียดชุดกราฟเส้นรายเดือนสำหรับทุกการเปรียบเทียบ (Every Comparison as a Monthly Line Chart)

1. **กราฟเส้น RMSE รายเดือน (Monthly RMSE Line Chart)**:
   - **เส้นเปรียบเทียบ**: ทั้ง 6 โมเดลหลักในรอบ $M_3$ เทียบกับ Raw NWP Baseline $B_0$
   - **การวิเคราะห์**: ตรวจสอบว่าโมเดลใดสามารถควบคุม RMSE ให้ต่ำได้อย่างคงเส้นคงวา โดยเฉพาะช่วงพีคมรสุมเดือน ส.ค. - ต.ค. ที่ความแปรปรวนของฝนสูงที่สุด
2. **กราฟเส้น MAE รายเดือน (Monthly MAE Line Chart)**:
   - **เส้นเปรียบเทียบ**: ค่า MAE รายเดือนของทุกโมเดลเทียบกับ $B_0$
   - **การวิเคราะห์**: สะท้อนระดับความคลาดเคลื่อนเฉลี่ยสภาวะปกติ (Typical Error) ในแต่ละช่วงฤดู
3. **กราฟเส้นอัตราส่วน RMSE/MAE รายเดือน (Monthly Error Ratio Line Chart)**:
   - **เส้นเปรียบเทียบ**: อัตราส่วน $\frac{\text{RMSE}}{\text{MAE}}$ รายเดือนของแต่ละโมเดล
   - **การวิเคราะห์**: ชี้วัดเดือนที่มี Extreme Outlier Misses รุนแรง (เช่น เม.ย. พายุฤดูร้อน หรือ ก.ย. ร่องมรสุมพาดผ่าน)
4. **กราฟเส้น Skill Scores รายเดือน (Monthly Skill Score Line Chart: $\text{SS}_{\text{RMSE}}, \text{SS}_{\text{MAE}}$)**:
   - **เส้นเปรียบเทียบ**: % การลด Error เทียบกับ Raw NWP รายเดือน ($\text{SS} > 0\%$)
   - **การวิเคราะห์**: พิสูจน์ว่า ML สามารถเพิ่มทักษะเหนือโมเดลสภาพอากาศโลกเดิมได้ทุกเดือนตลอดปี ไม่เกิดสภาวะ Negative Skill ในหน้าแล้ง
5. **กราฟเส้นความแม่นยำเตือนฝนหนักรายเดือน (Monthly Heavy Rain POD Line Chart @ $\ge 10, \ge 20$ mm/h)**:
   - **เส้นเปรียบเทียบ**: Probability of Detection (Hit Rate %) รายเดือนสำหรับเกณฑ์ฝนหนักและฝนหนักมาก
   - **การวิเคราะห์**: เปรียบเทียบความพร้อมของระบบเตือนภัยในเดือนวิกฤติน้ำหลาก
6. **กราฟเส้นอัตราเตือนหลอกรายเดือน (Monthly False Alarm Rate: FAR Line Chart)**:
   - **เส้นเปรียบเทียบ**: False Alarm Ratio (%) รายเดือน
   - **การวิเคราะห์**: ติดตามพฤติกรรมการเตือนเก้อ โดยเฉพาะในฤดูแล้ง (ม.ค. - ก.พ. และ พ.ย. - ธ.ค.) เพื่อลดผลกระทบ Cry Wolf Effect
7. **กราฟเส้นอัตราตกหนักแต่ไม่เตือนรายเดือน (Monthly Critical Miss Rate Line Chart)**:
   - **เส้นเปรียบเทียบ**: Critical Miss Rate (%) รายเดือน
   - **การวิเคราะห์**: ชี้วัดความปลอดภัยในชีวิตและทรัพย์สินของประชาชน ยืนยันว่าโมเดลไม่มองข้ามพายุฝนรุนแรงในเดือนสำคัญ
8. **กราฟเส้น Threat Score สุทธิรายเดือน (Monthly CSI Line Chart)**:
   - **เส้นเปรียบเทียบ**: Critical Success Index รายเดือน แยกตามเกณฑ์ $\ge 0.1, 2, 10, 20$ mm/h
   - **การวิเคราะห์**: แสดงดัชนีชี้วัดประสิทธิภาพรวมที่ลงโทษทั้งการเตือนหลอกและการไม่เตือน
9. **กราฟเส้นการพัฒนาตามรอบ Ablation รายเดือน (Monthly Ablation Progression Line Chart: $B_0 \to M_1 \to M_2 \to M_3$)**:
   - **เส้นเปรียบเทียบ**: เส้นแนวโน้ม 4 ระดับ ($B_0$, $M_1$, $M_2$, $M_3$) ในโมเดลเดียวกัน (เช่น ECMWF IFS)
   - **การวิเคราะห์**: พิสูจน์คุณค่าของการเพิ่ม Ground Obs ($M_1$) และดาวเทียม Himawari-9 ($M_3$) แยกรายเดือน (เช่น Himawari-9 โดดเด่นเป็นพิเศษในเดือนที่มีเมฆก่อยอดหนาแน่น)
10. **กราฟเส้น Generalization Gap รายเดือน (Monthly Generalization Gap Line Chart)**:
    - **เส้นเปรียบเทียบ**: $\Delta_{\text{Gen}}(\text{RMSE}) = \text{RMSE}(\text{DWR}) - \text{RMSE}(\text{HII})$ และ $\Delta_{\text{Gen}}(\text{MAE})$ รายเดือน
    - **การวิเคราะห์**: ตรวจสอบว่าโมเดลพยากรณ์บนสถานีอิสระที่ไม่เคยเห็น (DWR Hold-Out) ได้อย่างเสถียรเพียงใดตลอด 12 เดือน ไม่เกิดการทรุดตัวของประสิทธิภาพในฤดูใดฤดูหนึ่ง
11. **กราฟเส้น Lead-Time Degradation รายเดือน (Monthly Lead-Time Sensitivity Line Chart)**:
    - **เส้นเปรียบเทียบ**: เส้น Error ณ Lead Time +3h, +6h, +12h, +24h พล็อตต่อเนื่องตลอด 12 เดือน

---

## 5. แผนการตอบคำถามวิจัย (Research Questions Validation Matrix: Multi-Metric Focus)

ระบบรายงานผลสรุปจะประเมินผลการทดลองทั้ง **18 Configurations ($6 \text{ Models} \times 3 \text{ Variants}: M_1, M_2, M_3$)** เทียบกับ **$B_0$ Baseline** โดยเน้นการเปรียบเทียบ **RMSE, MAE, Seasonal Monthly Trends, และ Categorical Contingency Metrics** เพื่อตอบโจทย์วิจัยทั้ง 5 ข้อใน `plan.md` ข้อ 15 อย่างลึกซึ้ง:

| รหัสคำถาม | คำถามวิจัย (Research Question) | การวิเคราะห์เปรียบเทียบเชิงลึก (RMSE, MAE, Monthly Trends & Skill Scores) |
|---|---|---|
| **RQ1** | การทำ ML bias correction สามารถลด error ของ Weather Model ในไทยได้หรือไม่? | เปรียบเทียบ **$B_0$ (Raw Model) เทียบกับ $M_3$ ของทุกโมเดล**: วัด $\text{SS}_{\text{RMSE}}$ และ $\text{SS}_{\text{MAE}}$ ทั้งภาพรวมทั้งปีและกราฟเส้นรายเดือน (พิสูจน์ว่าลด Error ต่อเนื่องได้เฉลี่ยกี่ % ทั่วประเทศ) |
| **RQ2** | Spatial ground observations ช่วยปรับปรุง bias correction หรือไม่? | เปรียบเทียบผลต่าง **$M_1$ เทียบกับ $B_0$**: วิเคราะห์ว่าข้อมูลสถานีข้างเคียงช่วยลด RMSE และ MAE ได้เท่าใดในแต่ละระดับความสูง ภูมิประเทศ และช่วงฤดูกาล |
| **RQ3** | Himawari-9 ช่วยเพิ่มประสิทธิภาพเหนือ Weather Model + Ground หรือไม่? | เปรียบเทียบผลต่าง **$M_3$ เทียบกับ $M_1$**: พิสูจน์ผ่านกราฟเส้นรายเดือนว่าการรับรู้ยอดเมฆช่วยลด RMSE ในช่วงพายุฤดูร้อน (มี.ค.-เม.ย.) และฤดูมรสุม (Convective Outliers) ได้อย่างมีนัยสำคัญ |
| **RQ4** | ผลของ bias correction เปลี่ยนไปตาม lead time อย่างไร? | พล็อตกราฟเปรียบเทียบ **RMSE Degradation Curve, MAE Curve, และ CSI Curve** แยกตาม Lead Time 1h–24h ของทั้ง 6 โมเดลในรอบ $M_3$ ทั้งภาพรวมและรายเดือน |
| **RQ5** | ผลของ bias correction แตกต่างกันตามระดับความรุนแรงของฝนหรือไม่? | เปรียบเทียบความสามารถในการจับฝนหนัก ($\ge 10, \ge 20$ mm/h) เทียบกับฝนเบา: วัด Conditional RMSE/MAE เฉพาะช่วงฝนตกจริง และกราฟเส้น POD/FAR/Miss Rate ตลอด 12 เดือน ระหว่าง Single Tree (LGBM), Hurdle Model, และ Quantile ($q_{90}$) |

---

## 6. ผลลัพธ์และรายงานสรุป พร้อมตัวเลือก `--dir` (Deliverable Artifacts & CLI Specification)

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
│   ├── 01a_lead_time_rmse_curves.png           # [RMSE] กราฟเส้นเปรียบเทียบ RMSE vs Lead Time (1h-24h) ทุกโมเดล เทียบ Baseline B0
│   ├── 01b_lead_time_mae_curves.png            # [MAE] กราฟเส้นเปรียบเทียบ MAE vs Lead Time (1h-24h) ทุกโมเดล เทียบ Baseline B0
│   ├── 01c_rmse_mae_skill_score_barchart.png   # [Skill Score] Bar chart เปรียบเทียบ % การลด RMSE และ MAE ของทั้ง 18 โมเดล
│   ├── 01d_rmse_vs_mae_ratio_scatter.png       # [Error Ratio] Scatter plot RMSE vs MAE บ่งชี้สัดส่วน Extreme Error Outliers
│   ├── 01e_basin_rmse_mae_comparison.png       # [Spatial] เปรียบเทียบ RMSE และ MAE แยกตาม 25 ลุ่มน้ำสำคัญ
│   ├── 01f_generalization_gap_rmse_mae.png     # [Generalization] เปรียบเทียบ RMSE/MAE บน HII (In-Network) vs DWR (Blind Hold-Out)
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
│   ├── 10_national_2km_bias_map.png            # แผนที่ 2 km Grid แสดงผลการพยากรณ์และปรับแก้ทั่วประเทศ
│   ├── 11a_monthly_rmse_line_chart.png         # [รายเดือน] กราฟเส้นเปรียบเทียบ RMSE รายเดือน (Jan-Dec 2025) ทุกโมเดล vs B0
│   ├── 11b_monthly_mae_line_chart.png          # [รายเดือน] กราฟเส้นเปรียบเทียบ MAE รายเดือน (Jan-Dec 2025) ทุกโมเดล vs B0
│   ├── 11c_monthly_rmse_mae_ratio_line_chart.png # [รายเดือน] กราฟเส้นอัตราส่วน RMSE/MAE Ratio รายเดือน วินิจฉัย Extreme Outliers
│   ├── 11d_monthly_skill_scores_line_chart.png # [รายเดือน] กราฟเส้น Skill Scores (SS_RMSE, SS_MAE) รายเดือน
│   ├── 12a_monthly_heavy_rain_pod_line_chart.png # [รายเดือน] กราฟเส้นความแม่นยำเตือนฝนหนัก (POD Hit Rate) รายเดือน
│   ├── 12b_monthly_heavy_rain_far_line_chart.png # [รายเดือน] กราฟเส้นอัตราเตือนหลอก (False Alarm Ratio) รายเดือน
│   ├── 12c_monthly_heavy_rain_miss_rate_line_chart.png # [รายเดือน] กราฟเส้นอัตราตกหนักแต่ไม่เตือน (Critical Miss Rate) รายเดือน
│   ├── 12d_monthly_csi_line_chart.png          # [รายเดือน] กราฟเส้น Threat Score (CSI) รายเดือน ทุกระดับความรุนแรง
│   ├── 13_monthly_ablation_progression_line_chart.png # [รายเดือน] กราฟเส้นเปรียบเทียบ B0 vs M1 vs M2 vs M3 รายเดือน
│   ├── 14_monthly_generalization_gap_line_chart.png   # [รายเดือน] กราฟเส้น Generalization Gap (HII vs DWR Hold-Out) รายเดือน
│   └── 15_monthly_lead_time_rmse_mae_line_chart.png   # [รายเดือน] กราฟเส้นเปรียบเทียบ Lead-time Degradation แยกตามรายเดือน
├── predictions/                        # 3. จัดเก็บ CSV Result ผลการพยากรณ์สำหรับใช้ใน Benchmark อื่นๆ
│   ├── unified_benchmark_predictions_2025.csv  # รวมผลพยากรณ์ทุกโมเดลและสถานีตลอดปี 2025
│   ├── dwr_blind_predictions_2025.csv          # เฉพาะผลพยากรณ์บนเครือข่ายสถานี DWR Hold-Out
│   └── <model>_<ablation>_pred.csv             # แยกรายโมเดล (เช่น lgbm_m3_pred.csv, hurdle_m3_pred.csv)
│       # คอลัมน์มาตรฐาน: valid_time, run_time, lead_time, station_id, lat, lon, basin_id,
│       #                 observed_rain, nwp_raw_rain, predicted_bias, corrected_rain, model_name
└── reports/                            # 4. สรุปรายงานตัวเลขและตารางสถิติ
    ├── final_benchmark_report_2025.md          # รายงานสรุปภาพรวมและคำตอบ RQ1–RQ5
    ├── rmse_mae_cross_model_benchmark.csv      # [หลัก] ตารางสรุป RMSE, MAE, Ratio, และ Skill Scores (SS_RMSE, SS_MAE) ครบ 18 Runs + B0
    ├── lead_time_rmse_mae_breakdown.csv        # [หลัก] ตารางแจกแจงค่า RMSE และ MAE แยกตาม Lead Time (+1h ถึง +24h)
    ├── monthly_rmse_mae_benchmark.csv          # [หลัก] ตารางแจกแจงค่า RMSE, MAE, Ratio และ Skill Score แยกราย 12 เดือน (Jan-Dec 2025)
    ├── monthly_contingency_benchmark.csv       # [หลัก] ตารางแจกแจงสถิติเตือนภัย (POD, FAR, Miss Rate, CSI) แยกราย 12 เดือน
    ├── monthly_generalization_gap.csv          # [หลัก] ตารางแจกแจง Generalization Gap (HII vs DWR) แยกราย 12 เดือน
    ├── basin_rmse_mae_breakdown.csv            # [หลัก] ตารางแจกแจงค่า RMSE และ MAE แยกรายลุ่มน้ำ
    ├── generalization_gap_report.csv           # [หลัก] ตารางเปรียบเทียบ Error Gap (RMSE, MAE) ระหว่างสถานี HII vs DWR Hold-Out
    ├── rain_intensity_contingency_table.csv    # สถิติ CSI, POD, FAR แยกรายระดับฝน
    ├── model_ablation_matrix_18runs.csv        # สรุปผล 18 Runs (6 Models x 3 Variants) + B0
    ├── heavy_rain_warning_stats.csv            # สถิติ POD, FAR, Missed Count สำหรับฝน >=10 และ >=20 mm/h
    └── basin_skill_scores.csv                  # สถิติเปรียบเทียบแยกรายลุ่มน้ำ
```

---

## 7. สรุป Checklist ความพร้อม Phase 5

- [ ] ออกแบบฟังก์ชันคำนวณและเปรียบเทียบ Continuous Metrics (RMSE, MAE, RMSE/MAE Ratio, Skill Scores $\text{SS}_{\text{RMSE}}, \text{SS}_{\text{MAE}}$)
- [ ] ออกแบบฟังก์ชันคำนวณ Categorical Contingency Metrics (CSI, POD, FAR, F1 @ $\ge 0.1, 2, 10, 20$ mm/h)
- [ ] กำหนดกระบวนการจับคู่ Spatial Verification ระหว่าง 2 km Grid กับ Ground Truth Stations ปี 2025
- [ ] ออกแบบตารางแจกแจงการเปรียบเทียบ RMSE และ MAE ตาม Lead Time (+1h ถึง +24h)
- [ ] ออกแบบชุดกราฟเส้นเปรียบเทียบรายเดือน (12-Month Line Charts Suite) ครอบคลุมทุกการเปรียบเทียบ (RMSE, MAE, Ratio, Skill Score, POD, FAR, Miss Rate, CSI, Ablation, Generalization Gap)
- [ ] ออกแบบตารางสรุปผลรายเดือน (`monthly_rmse_mae_benchmark.csv`, `monthly_contingency_benchmark.csv`, `monthly_generalization_gap.csv`)
- [ ] ออกแบบตารางและกราฟเปรียบเทียบ Generalization Gap (RMSE & MAE บน HII vs DWR Hold-Out)
- [ ] จัดวางเทมเพลตสำหรับเขียนสรุปผลงานวิจัยเพื่อตอบครบทั้ง 5 RQs (RQ1–RQ5) พร้อมหลักฐานเชิงประจักษ์จาก RMSE และ MAE ตลอดทั้ง 12 เดือน
