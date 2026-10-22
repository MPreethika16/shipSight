# ShipSight: Explainable ETA and Multi-Stage Delay Intelligence

[![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

> **Author:** Preethika Manepally  
> **Evaluation Scope:** Aiqon AI Internship Assessment (Round 2)  
> Predicting shipment arrival times, quantifying cascading delay propagation across multi-stage logistics lifecycles, and providing feature attribution at the vessel berthing cutoff.

---

## 1. Project Overview & Problem Framing

In international intermodal logistics, estimated time of arrival (ETA) is rarely static. Delays occurring at one stage—such as bad weather during maritime transit, anchorage congestion at destination ports, or physical customs holds—frequently compound downstream, disrupting truck scheduling, warehouse labor, and container demurrage budgets.

**ShipSight** models these cascading dynamics across a 5-stage shipment lifecycle:
1. **Stage 1 (Maritime Transit):** Origin port departure to destination anchorage.
2. **Stage 2 (Port Anchorage & Berthing):** Pilot boarding, tug assistance, and berthing.
3. **Stage 3 (Terminal Container Handling):** Vessel discharge, crane moves, and yard dwell.
4. **Stage 4 (Customs Clearance):** Automated release vs. physical non-intrusive inspection (NII).
5. **Stage 5 (Inland Drayage):** Gate-out haulage to final consignee delivery.

---

## 2. Why Option B (Synthetic Multi-Stage Simulation)?

### Exploratory AIS Telemetry Review & Data Provenance Notice
Prior to architecture selection, an exploratory review of real-world Automatic Identification System (AIS) vessel telemetry was conducted to assess whether public feeds (such as NOAA MarineCadastre coastal archives and AISHub position snapshots) could support full supply chain ETA modeling.

> **Provenance Notice:** The raw exploratory AIS extract files (previously referenced under working titles such as `AIS_2017_01_Zone01.csv` and `aishub_snapshot_10k.json`) and preliminary audit scripts are not included in this submission repository. Consequently, specific sample counts and telemetry artifact percentages from that initial investigation represent exploratory findings from an earlier scoping review that are not directly reproducible from the submitted archive.

Regardless of specific sample figures, the exploratory review confirmed three fundamental methodological bottlenecks that make raw AIS data insufficient on its own for multi-stage intermodal ETA prediction:
1. **Zero Visibility into Landside & Terminal Operations:** AIS transponders transmit marine positions (latitude, longitude, speed over ground, course over ground). They contain no records of shore-side milestones (vessel discharge, customs clearance, terminal container dwell, or inland gate-out delivery).
2. **Absence of Baseline Schedules:** Transponder broadcasts do not contain contractual carrier booking schedules ($T_{\text{sched}}$), making delay calculation ($\text{Actual} - \text{Scheduled}$) impossible without external booking systems.
3. **Telemetry & Attribute Quality Gaps:** Broadcast transponder streams regularly exhibit noisy or unpopulated destination strings, unavailable heading sentinels (`511`), and spatial clipping across coastal receiver zones.

Following Aiqon's assignment guidelines (*"You may use publicly available datasets or construct a synthetic dataset... A dataset of approximately 100–300 shipment records is sufficient"*), we designed **Option B**: a sequence-consistent, physics-grounded synthetic benchmark of **$N = 250$ shipments** across 5 global trade corridors.

---

## 3. Mathematical Formulation

### Sequential Timestamp Invariant
Each stage $k$ has its scheduled duration $d_k^{\text{sched}}$ and actual duration $d_k^{\text{actual}}$.
- **Scheduled Baseline:** $T_k^{\text{sched}} = T_{k-1}^{\text{sched}} + d_k^{\text{sched}}$ starting from $T_0^{\text{sched}}$.
- **Actual Realization:** $T_0^{\text{actual}} = T_0^{\text{sched}} + \Delta t_{\text{dep}}$ and $T_k^{\text{actual}} = T_{k-1}^{\text{actual}} + d_k^{\text{actual}}$.

Departure delay is decoupled from waterborne transit duration ($d_1^{\text{actual}}$), ensuring zero double-counting. Total final delay satisfies:
$$\text{final\_delay\_hours} = \frac{T_5^{\text{actual}} - T_5^{\text{sched}}}{3600} = \Delta t_{\text{dep}} + \sum_{k=1}^5 \left( d_k^{\text{actual}} - d_k^{\text{sched}} \right)$$

### Operational Prediction Cutoff ($t_{\text{pred}}$)
$$\mathbf{t_{\text{pred}} = \text{actual\_berth\_time} = T_2^{\text{actual}}}$$
When the ship docks, terminal operators and consignees require a reliable forecast to book drayage trucks and warehouse labor. All downstream durations ($d_3, d_4, d_5$) and timestamps ($T_3, T_4, T_5$) are strictly excluded to prevent lookahead target leakage.

### Prediction Targets
1. **Primary Target (Remaining Hours):**
   $$y_{\text{rem}} = d_3^{\text{actual}} + d_4^{\text{actual}} + d_5^{\text{actual}}$$
   $$\widehat{\text{ETA}} = T_2^{\text{actual}} + \hat{y}_{\text{rem}}$$
2. **Derived Final Delay:**
   $$\hat{y}_{\text{delay}} = \frac{T_2^{\text{actual}} + \hat{y}_{\text{rem}} - T_5^{\text{sched}}}{3600}$$
3. **Derived Delay Risk:**
   $$\hat{y}_{\text{risk}} = \mathbb{I}(\hat{y}_{\text{delay}} > 6.0\text{ hours})$$

---

## 4. Leakage Prevention: Centralized Whitelist

To eliminate lookahead bias, all downstream actual durations ($d_3, d_4, d_5$), downstream timestamps ($T_3, T_4, T_5$), and target derivatives are strictly prohibited from model inputs.

### Admitted Features (`FEATURE_WHITELIST`)
- **Corridor & Shipment Metadata:** `route_id`, `carrier_code`, `cargo_type`, `container_teu`, `transit_distance_nm`
- **Scheduled Buffers:** `scheduled_terminal_hours`, `scheduled_customs_hours`, `scheduled_drayage_hours`
- **Upstream Realizations (Known at $t_{\text{pred}}$):** `weather_severity_index`, `carried_delay_at_anchorage`, `port_congestion_index`, `carried_delay_at_berth`, `customs_inspection_flag`

#### Operational Assumption on `customs_inspection_flag`
In the benchmark simulation, `customs_inspection_flag` represents a pre-arrival regulatory inspection decision assumed to be known by the time the vessel berths (e.g., automated electronic manifest screening via CBP ACE / Automated Targeting System prior to port arrival). This feature represents an assumed, pre-known administrative decision, **not an actual inspection outcome or downstream customs event**. In an operational deployment, this feature is valid only if regulatory hold decisions are genuinely made available to the model at or before vessel docking. Real-world availability was not empirically verified against live customs APIs.

---

## 5. Repository Structure

```text
shipsight/
├── data/
│   └── synthetic_shipments.csv       # 250-row generated benchmark
├── src/
│   ├── __init__.py
│   ├── config.py                     # Corridors, whitelist, seeds, thresholds
│   ├── generator.py                  # Simulation engine & sequential timestamps
│   ├── model.py                      # Baseline, HistGradientBoosting, evaluation
│   └── propagation_analysis.py       # Stage delay transmission analytics
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # 8 automated unit tests
├── run_pipeline.py                   # Master end-to-end execution runner
├── convert_report_to_pdf.py          # Standalone headless PDF generator
├── requirements.txt                  # Python package dependencies
├── SHIP_SIGHT_TECHNICAL_REPORT.md    # 2-3 page assignment technical report
├── SHIP_SIGHT_TECHNICAL_REPORT.pdf   # 3-page submission PDF report
└── README.md                         # Project documentation
```

---

## 6. Quickstart & Reproducibility

### Installation
Ensure Python 3.12+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Run End-to-End Pipeline
Executes data generation, unit tests, model training, evaluation, and delay-propagation analysis:
```bash
python run_pipeline.py
```

### Run Test Suite Independently
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## 7. Empirical Results (Chronological Test Set, $N = 50$)

| Evaluation Metric | Scheduled Baseline | HistGradientBoosting | Majority-Class (Always On-Time) |
| :--- | :--- | :--- | :--- |
| **MAE (hours)** | **10.608 hrs** | **5.921 hrs (-44.2%)** | — |
| **RMSE (hours)** | **15.853 hrs** | **7.237 hrs (-54.3%)** | — |
| **MAPE (%)** *(Caveat: zero-skew)* | 14.17% | 8.89% | — |
| **Delay Risk Precision ($\tau=6\text{h}$)** | 0.846 | 0.882 | 0.000 |
| **Delay Risk Recall ($\tau=6\text{h}$)** | 0.550 | 0.750 | 0.000 |
| **Delay Risk F1-Score** | 0.667 | 0.811 | 0.000 |

*Baseline Definition:* The scheduled-duration baseline assumes that downstream stages finish within their scheduled allowances ($d_3^{\text{sched}} + d_4^{\text{sched}} + d_5^{\text{sched}}$).

### Confusion Matrices ($\tau = 6.0\text{ hours}$, Labels: [0 = On-Time, 1 = Delayed])
- **Test Set Ground Truth:** 30 On-Time (60%), 20 Delayed (40%).
- **Scheduled Baseline:** `[[28, 2], [9, 11]]` (misclassified 9 delayed shipments as on-time; Recall = 55.0%).
- **HistGradientBoosting:** `[[28, 2], [5, 15]]` (reduced false negatives to 5, achieving Recall = 75.0% and F1 = 0.811).
- **Majority-Class Benchmark:** `[[30, 0], [20, 0]]` (Accuracy: 60.0%, Precision: 0.0, Recall: 0.0, F1: 0.000).

*Evaluation Summary:* The model achieved a higher delay-risk F1-score than the scheduled-duration baseline on the held-out synthetic test set (0.811 vs 0.667).

### Permutation Feature Importance (MAE Degradation When Shuffled)
1. `customs_inspection_flag`: **+8.870 hrs**
2. `transit_distance_nm`: **+4.252 hrs**
3. `route_id`: **+1.638 hrs**
4. `container_teu`: **+0.053 hrs**
5. `port_congestion_index`: **+0.048 hrs**

### Stage-to-Stage Delay Transmission Dynamics
- **Departure $\rightarrow$ Anchorage:** Transmission slope = **$0.555$** (schedule allowances and speed adjustments absorb departure delays).
- **Anchorage $\rightarrow$ Berthing:** Transmission slope = **$1.053$** (queue compounding when vessels arrive off-schedule).
- **Berthing $\rightarrow$ Terminal:** Transmission slope = **$1.008$** (container terminal turnaround compounding).
- **Customs Inspection Penalty:** Inspected shipments average **$52.66\text{ hrs}$** vs. **$16.01\text{ hrs}$** for automated electronic release (Net disruption penalty: **$+36.65\text{ hrs}$**).

---

## 8. Explicit Assumptions, Boundary Disclosures & Limitations

1. **Synthetic Nature:** All records are simulated using parametric distributions (Normal, Uniform, Beta, Exponential). The dataset is synthetic and must not be interpreted as validated performance on live maritime carrier telematics.
2. **Attribution vs. Causality:** Permutation importance reflects the statistical sensitivity of the trained GBDT estimator on 200 training rows, not proven physical causality in an operational terminal.
3. **Customs Availability Assumption:** `customs_inspection_flag` assumes pre-arrival screening notifications are accessible at berth; if unavailable in practice, downstream prediction error will increase.
4. **Small Sample Benchmark ($N = 250$):** Sized strictly for the take-home assessment. In production environments, multi-year datasets capturing seasonal peaks are necessary.
5. **Single-Leg Simplification:** The model does not simulate transshipment hub restowage or feeder vessel missed connections.
6. **Exploratory AIS Audit Provenance:** Raw exploratory AIS files and preliminary audit scripts are omitted from this submission repository to maintain a self-contained archive; preliminary audit findings are preserved solely for methodological rationale and are not reproducible from this repository.
