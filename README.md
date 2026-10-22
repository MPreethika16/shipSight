# ShipSight: Explainable ETA and Multi-Stage Delay Intelligence

[![CI](https://github.com/MPreethika16/shipSight/actions/workflows/ci.yml/badge.svg)](https://github.com/MPreethika16/shipSight/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)]()
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

### Prediction Targets & Formulations
1. **Direct Regression Target (Remaining Duration):**
   $$y_{\text{rem}} = d_3^{\text{actual}} + d_4^{\text{actual}} + d_5^{\text{actual}}, \quad \widehat{\text{ETA}} = T_2^{\text{actual}} + \hat{y}_{\text{rem}}$$
2. **Residual Regression Target (`ResidualETAPredictor`):**
   $$\Delta_{\text{rem}} = y_{\text{rem}} - (d_3^{\text{sched}} + d_4^{\text{sched}} + d_5^{\text{sched}})$$
   $$\hat{y}_{\text{rem}} = \max\left(0.0, \; (d_3^{\text{sched}} + d_4^{\text{sched}} + d_5^{\text{sched}}) + \widehat{\Delta}_{\text{rem}}\right)$$
   *Advantage:* Expresses downstream durations as deviations from scheduled allowances ($\Delta_{\text{rem}} \approx [-4, +40\text{h}]$), mitigating tree leaf bounding on unseen routes within this synthetic benchmark.
3. **Derived Final Delay & Delay Risk:**
   $$\hat{y}_{\text{delay}} = \frac{T_2^{\text{actual}} + \hat{y}_{\text{rem}} - T_5^{\text{sched}}}{3600}, \quad \hat{y}_{\text{risk}} = \mathbb{I}(\hat{y}_{\text{delay}} > 6.0\text{ hours})$$

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
├── .github/
│   └── workflows/
│       └── ci.yml                    # Automated GitHub Actions test & smoke test
├── data/
│   └── synthetic_shipments.csv       # 250-row generated benchmark
├── src/
│   ├── __init__.py
│   ├── config.py                     # Corridors, whitelist, seeds, thresholds
│   ├── generator.py                  # Simulation engine & sequential timestamps
│   ├── model.py                      # Baseline, Direct GBDT, ResidualETAPredictor
│   ├── propagation_analysis.py       # Stage delay transmission analytics
│   ├── corridor_evaluation.py        # Leave-one-corridor-out cross-validation
│   ├── shipment_explainer.py         # Individual shipment prediction explanations
│   ├── ablation_and_scenarios.py     # Customs feature ablation & disruption analysis
│   └── repeated_seed_evaluation.py   # Multi-seed stability evaluation (10 seeds)
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py              # Core pipeline unit tests (8 tests)
│   ├── test_residual_model.py        # Residual target, clipping, leakage & backwards compat (9 tests)
│   ├── test_corridor_evaluation.py   # Corridor isolation & LOCO tests (3 tests)
│   ├── test_shipment_explainer.py    # Live vs retrospective modes, scenario sensitivity (9 tests)
│   ├── test_ablation_and_scenarios.py# Ablation & scenario reproducibility (3 tests)
│   └── test_repeated_seeds.py        # Multi-seed determinism & win count tests (4 tests)
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
Ensure Python 3.11 or 3.12 is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Run End-to-End Pipeline
Executes data generation, unit tests, model training, repeated-seed evaluation, and delay-propagation analysis:
```bash
python run_pipeline.py
```

### Run Test Suite Independently
Executes all 36 automated unit tests across core pipeline, residual model, corridor isolation, shipment explainer, disruption scenarios, and repeated seeds:
```bash
python -m pytest -v
```

### Run Extended Evaluation Modules
1. **Repeated-Seed Stability Evaluation (10 Seeds):**
   ```bash
   python -m src.repeated_seed_evaluation
   ```
2. **Leave-One-Corridor-Out (LOCO) Generalization:**
   ```bash
   python -m src.corridor_evaluation
   ```
3. **Customs Feature Ablation & Disruption Scenarios:**
   ```bash
   python -m src.ablation_and_scenarios
   ```
4. **Inspect Individual Shipment Explanation (Live Forecast & Retrospective Audit):**
   ```bash
   python -c "from src.generator import generate_synthetic_shipments; from src.model import split_chronological, build_feature_matrix, build_pipeline; from src.shipment_explainer import explain_shipment, format_shipment_explanation; import pandas as pd; records = generate_synthetic_shipments(250, 42); train, test = split_chronological(records); X, y, _, _ = build_feature_matrix(train); p = build_pipeline().fit(pd.DataFrame(X), y); print('=== LIVE MODE ===\n', format_shipment_explanation(explain_shipment(test[0], p, mode='live'))); print('\n=== RETROSPECTIVE MODE ===\n', format_shipment_explanation(explain_shipment(test[0], p, mode='retrospective')))"
   ```

---

## 7. Empirical Results

### 7.1 Primary Chronological Benchmark ($N_{\text{test}} = 50$)

| Evaluation Metric | Scheduled Baseline | Direct GBDT | Residual GBDT | Majority-Class | Res. vs. Base |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAE (hours)** | 10.608 hrs | 5.921 hrs | **5.414 hrs** | — | **-49.0%** |
| **RMSE (hours)** | 15.853 hrs | 7.237 hrs | **6.578 hrs** | — | **-58.5%** |
| **MAPE (%)** *(Caveat: zero-skew)* | 14.17% | 8.89% | **8.09%** | — | -42.9% |
| **Delay Risk Precision ($\tau=6\text{h}$)** | 0.846 | **0.882** | **0.882** | 0.000 | +4.3% |
| **Delay Risk Recall ($\tau=6\text{h}$)** | 0.550 | **0.750** | **0.750** | 0.000 | **+36.4%** |
| **Delay Risk F1-Score** | 0.667 | **0.811** | **0.811** | 0.000 | **+21.6%** |

*Baseline Definition:* The scheduled-duration baseline assumes that downstream stages finish within their scheduled allowances ($d_3^{\text{sched}} + d_4^{\text{sched}} + d_5^{\text{sched}}$).

#### Confusion Matrices ($\tau = 6.0\text{ hours}$, Labels: [0 = On-Time, 1 = Delayed])
- **Test Set Ground Truth:** 30 On-Time (60%), 20 Delayed (40%).
- **Scheduled Baseline:** `[[28, 2], [9, 11]]` (misclassified 9 delayed shipments as on-time; Recall = 55.0%).
- **Direct & Residual GBDT:** `[[28, 2], [5, 15]]` (reduced false negatives to 5, achieving Recall = 75.0% and F1 = 0.811).
- **Majority-Class Benchmark:** `[[30, 0], [20, 0]]` (Accuracy: 60.0%, Precision: 0.0, Recall: 0.0, F1: 0.000).

#### Permutation Feature Importance (MAE Degradation When Shuffled)
1. `customs_inspection_flag`: **+8.870 hrs**
2. `transit_distance_nm`: **+4.252 hrs**
3. `route_id`: **+1.638 hrs**
4. `container_teu`: **+0.053 hrs**
5. `port_congestion_index`: **+0.048 hrs**

#### Stage-to-Stage Delay Transmission Dynamics
- **Departure $\rightarrow$ Anchorage:** Transmission slope = **$0.555$** (schedule allowances and speed adjustments absorb departure delays).
- **Anchorage $\rightarrow$ Berthing:** Transmission slope = **$1.053$** (queue compounding when vessels arrive off-schedule).
- **Berthing $\rightarrow$ Terminal:** Transmission slope = **$1.008$** (container terminal turnaround compounding).
- **Customs Inspection Penalty:** Inspected shipments average **$52.66\text{ hrs}$** vs. **$16.01\text{ hrs}$** for automated electronic release (Net disruption penalty: **$+36.65\text{ hrs}$**).

### 7.2 Leave-One-Corridor-Out (LOCO) Generalization Benchmark
To evaluate model transfer to completely unseen trade corridors, each corridor is held out in turn while the pipeline is trained exclusively on the remaining 4 corridors:

| Held-Out Corridor | N | Baseline MAE | Direct GBDT MAE | Residual GBDT MAE | Residual vs. Base |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BEANR-USNYC** (3,450 nm) | 50 | 10.769 h | 10.766 h (+0.0%) | **5.715 h** | **+46.9%** |
| **CNSHA-USLAX** (5,700 nm) | 50 | 11.037 h | 8.531 h (+22.7%) | **4.772 h** | **+56.8%** |
| **INNSA-GBFXT** (6,400 nm) | 50 | 11.978 h | 6.756 h (+43.6%) | **5.513 h** | **+54.0%** |
| **KRPUS-JPYOK** (620 nm feeder) | 50 | 8.578 h | 20.779 h (-142.2%) | **4.459 h** | **+48.0%** |
| **SGSIN-NLRTM** (8,300 nm) | 50 | 9.186 h | 10.585 h (-15.2%) | **5.714 h** | **+37.8%** |
| **Macro Average** | 250 | 10.310 h | 11.483 h (-11.4%) | **5.235 h** | **+49.2%** |

*Root Cause Analysis & Residual Resolution:*
- **The Deficit in Direct GBDT:** Direct GBDT suffered from negative transfer on the regional feeder lane `KRPUS-JPYOK` (-142.2% error surge), dragging macro MAE to 11.483h (-11.4% worse than baseline). Because tree ensembles cannot predict below the lowest target values seen in training ($y_{\text{train}} \in [55, 115\text{h}]$ vs. feeder allowance $44\text{h}$), the model suffered from leaf bounding bias.
- **The Residual Resolution:** Reformulating the target as scheduled residual $\Delta_{\text{rem}} = y_{\text{actual}} - y_{\text{sched}}$ centers predictions around route-specific scheduled allowances ($\Delta_{\text{rem}} \approx [-4, +40\text{h}]$ in this benchmark). On this synthetic five-corridor experiment, Residual GBDT overcame the feeder extrapolation deficit observed with direct trees, outperforming the scheduled baseline across **all 5 out of 5 corridors (100% win rate)** with a Macro MAE of **5.235h (+49.2% improvement)** and Macro RMSE of **6.778h (+58.7% improvement)**. *Note: These results reflect observed behavior on this synthetic 5-corridor simulation and do not constitute a universal generalization guarantee on live maritime networks.*

### 7.3 Customs Feature Ablation & Controlled Disruption Scenarios
- **Feature Ablation on `customs_inspection_flag`:** Removing the advance customs hold indicator increases test MAE from **5.921h to 11.599h (+95.9%)** and RMSE from **7.237h to 15.766h (+117.9%)**, causing the model to perform worse than the scheduled baseline (10.608h MAE). This confirms that advance regulatory notice is the dominant determinant of terminal dwell variance.
- **Input-Sensitivity Stress Test:** Elevating `port_congestion_index` by +0.35 at inference time (holding ground truth constant) shifts remaining-duration forecasts by an average of **+0.123 hours**.
- **Ground-Truth Regulatory Hold Disruption:** Imposing an unannounced +24.0h physical hold on inspected test containers increases model test MAE from **5.921h to 7.557h** (while scheduled baseline MAE spikes to **14.928h**).

### 7.4 Shipment-Level Prediction Explanations & Model Sensitivity
The explainer engine (`src/shipment_explainer.py`) provides transparent, auditable shipment-level intelligence with strict separation of operational contexts:

- **Live Operational Prediction Mode (`mode="live"`):**
  - Designed for real-time dispatch and demurrage planning at vessel berthing ($t_{\text{pred}} = T_2^{\text{actual}}$).
  - Reports prediction cutoff, observed upstream delay accumulated ($D_2$), scheduled remaining allowance, predicted remaining duration, predicted final ETA, and predicted delay relative to scheduled delivery.
  - **Zero Downstream Leakage:** Realized downstream delivery timestamps, actual final delay, dwell durations, and forecast errors are strictly quarantined and never revealed.
- **Retrospective Evaluation Audit Mode (`mode="retrospective"`):**
  - Designed for post-voyage performance auditing and carrier scorecards.
  - Compares operational predictions against ground-truth realized milestones (`actual_final_delivery_time`, `final_delay_hours`) and reports exact forecast errors.
- **Fitted Model Scenario Sensitivity:**
  - Evaluates what-if variations by re-running the actual fitted regression pipeline on modified copies of admissible features at the berthing cutoff (e.g., counterfactually toggling `customs_inspection_flag: 0 <-> 1` or simulating port congestion surges).
  - **Methodological Disclaimer:** Sensitivity shifts reflect response-surface variations of the fitted statistical estimator. They are **explicitly NOT physical causal attributions, NOT guarantees of real-world outcome interventions, and MUST NOT be labeled as additive SHAP values**.

### 7.5 Repeated-Seed Stability Evaluation (10 Deterministic Runs)
To evaluate algorithmic stability across stochastic variations in synthetic journey realization, the pipeline was evaluated across 10 deterministic random seeds ($N=250$ shipments per run, 80/20 chronological split, total 2,500 simulated shipments):

| Metric | Scheduled Baseline | Direct GBDT | Residual GBDT | Res. vs. Base |
| :--- | :---: | :---: | :---: | :---: |
| **MAE (hours)** | 10.643 +/- 1.928 hrs | 4.915 +/- 0.608 hrs | **4.678 +/- 0.405 hrs** | **+55.2%** (std: 5.8%) |
| **RMSE (hours)** | 16.738 +/- 2.491 hrs | 6.436 +/- 0.798 hrs | **6.054 +/- 0.464 hrs** | **+63.3%** (std: 4.1%) |
| **Delay Risk F1 ($	au=6\text{h}$)** | 0.706 +/- 0.094 | 0.883 +/- 0.046 | **0.894 +/- 0.045** | **+26.6%** |
| **Delay Risk Precision** | 0.846 +/- 0.082 | 0.906 +/- 0.042 | **0.907 +/- 0.041** | — |
| **Delay Risk Recall** | 0.612 +/- 0.115 | 0.866 +/- 0.081 | **0.886 +/- 0.076** | — |

- **Win Count:** Both Direct and Residual GBDT achieved lower MAE than the scheduled baseline in **10 out of 10 runs (100.0%)**. Residual GBDT exhibited tighter standard deviations and superior generalization across every seed.
- **Improvement Formula & Convention:** Reductions are evaluated per seed as `((baseline - model) / baseline) * 100`. For MAE, Residual GBDT achieves a mean per-seed reduction of **+55.2% (std: 5.8%)** (Direct: **+53.0% (std: 6.5%)**), while aggregate mean MAE yields 56.0% and 53.8%. For RMSE, Residual GBDT achieves a mean per-seed reduction of **+63.3% (std: 4.1%)** (Direct: **+61.1% (std: 5.1%)**), while aggregate mean RMSE yields 63.8% and 61.5%.
- **Model MAE Range:** Minimum MAE = 3.968 hrs, Maximum MAE = 5.921 hrs (Seed 42).
- **Seeds Evaluated:** `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`.

---

## 8. Explicit Assumptions, Boundary Disclosures & Limitations

1. **Synthetic Nature:** All records are simulated using parametric distributions (Normal, Uniform, Beta, Exponential). The dataset is synthetic and must not be interpreted as validated performance on live maritime carrier telematics.
2. **Attribution vs. Causality:** Permutation importance reflects the statistical sensitivity of the trained GBDT estimator on 200 training rows, not proven physical causality in an operational terminal.
3. **Customs Availability Assumption:** `customs_inspection_flag` assumes pre-arrival screening notifications are accessible at berth; if unavailable in practice, downstream prediction error will increase.
4. **Small Sample Benchmark ($N = 250$):** Sized strictly for the take-home assessment. In production environments, multi-year datasets capturing seasonal peaks are necessary.
5. **Single-Leg Simplification:** The model does not simulate transshipment hub restowage or feeder vessel missed connections.
6. **Exploratory AIS Audit Provenance:** Raw exploratory AIS files and preliminary audit scripts are omitted from this submission repository to maintain a self-contained archive; preliminary audit findings are preserved solely for methodological rationale and are not reproducible from this repository.
7. **Unseen Route Transfer Boundary:** Model transfer between disparate route topologies (long-haul transoceanic vs. short-sea regional feeders) requires lane-specific normalization or stratified corridor training.
8. **Scenario Analysis Scope:** Controlled scenario evaluations reflect deterministic manipulations of the synthetic simulation engine and must not be interpreted as empirical causal effects established from real-world maritime operations.
9. **Parametric Stability Scope:** Repeated-seed evaluations measure algorithmic stability under the parameterized generative equations across pseudo-random generator states; they do not establish empirical statistical significance on real-world shipping telematics.
10. **Model Sensitivity vs. Causal Attribution:** What-if scenario sensitivity reflects the statistical response-surface derivative of the fitted machine learning model on modified feature vectors; it does not claim physical causal mechanisms or guarantee real-world intervention outcomes.
