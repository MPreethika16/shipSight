# Technical Report: ShipSight — Explainable ETA and Delay Intelligence

**Author:** Preethika Manepally  
**Project:** ShipSight — Multi-Stage Shipment ETA and Delay Propagation  
**Evaluation Scope:** Aiqon AI Internship Assessment (Round 2)  
**Date:** October 2026  

---

## 1. Executive Summary

In international containerized shipping, an Estimated Time of Arrival (ETA) is an evolving operational signal rather than a static forecast. Disruptions occurring at upstream stages—such as adverse weather during maritime navigation or anchorage queuing outside congested ports—frequently cascade into downstream terminal handling, customs holds, and inland delivery delays.

This report presents **ShipSight**, an explainable machine learning system developed for Aiqon to:
1. Forecast remaining journey duration and final ETA at the critical operational decision point: **vessel berthing** ($t_{\text{pred}} = \text{actual\_berth\_time}$).
2. Estimate shipment delay magnitude ($\text{final\_delay\_hours}$) and classify delay risk against an explicit operational threshold ($\tau = 6.0\text{ hours}$).
3. Model and quantify how delays accumulate, amplify, or absorb across a 5-stage shipment lifecycle.
4. Provide rigorous feature attribution without introducing temporal lookahead leakage.

Evaluating on a strictly chronological held-out test set ($N=50$, 60% on-time vs. 40% delayed), our direct `HistGradientBoostingRegressor` achieves an **MAE of 5.921 hours** and **RMSE of 7.237 hours** (44.2% MAE reduction over the scheduled baseline at 10.608 hours). Our residual-learning formulation ($\text{residual} = y_{\text{actual}} - y_{\text{sched}}$, $\hat{y} = \max(0, y_{\text{sched}} + \hat{\text{residual}})$) further improves holdout performance to **5.414 hours MAE** (**-49.0% vs. baseline**, -8.6% vs. direct model) and **6.578 hours RMSE** (**-58.5% vs. baseline**). On the derived delay-risk classification task ($\tau = 6.0\text{h}$), both ML models achieve an F1-score of **0.811** (Precision: 0.882, Recall: 0.750), significantly outperforming the scheduled baseline (0.667 F1) and a majority-class benchmark (0.000 F1). Furthermore, within this synthetic five-corridor benchmark, residual learning resolved the direct model's out-of-distribution extrapolation deficit on the evaluated short-sea feeder lane.

---

## 2. Exploratory Data Review & Methodology Decision

Prior to architecture selection, an exploratory investigation into real-world Automatic Identification System (AIS) vessel telemetry was conducted using public extracts (including NOAA MarineCadastre coastal archives and AISHub vessel position snapshots).

*Provenance Disclosure:* The raw exploratory AIS sample files and preliminary audit scripts are not included in this submitted archive to maintain a lightweight, self-contained benchmark repository. Specific exploratory sample sizes and error-rate statistics cited during initial scoping represent findings from an earlier review that are not directly reproducible from the submitted archive.

### Methodological Limitations of Raw AIS Telemetry
1. **Absence of Supply Chain Lifecycle Events:** Raw AIS captures marine transponder fixes ($\text{LAT}, \text{LON}$) and vessel kinematics ($\text{SOG}, \text{COG}$). It possesses **zero visibility into landside and terminal operations** (berth discharge, customs clearance, terminal yard dwell, inland gate-out).
2. **Missing Baseline Schedules:** Raw AIS broadcasts do not contain contractual carrier schedules ($T_{\text{sched}}$). Consequently, delay ($\text{Actual} - \text{Scheduled}$) cannot be calculated without external booking schedules.
3. **Telemetry & Attribute Quality Gaps:** Broadcast AIS feeds routinely exhibit corrupted kinematic reports, unavailable heading sentinels (`511`), unstandardized or missing free-text destination strings, and spatial receiver clipping that fractures journey continuity.

### The Decision for Option B (Structured Multi-Stage Benchmark)
Because Aiqon's problem statement explicitly requires modeling **delay propagation across intermediate stages** (port arrival, customs clearance, final delivery) and authorizes constructing a synthetic dataset of 100–300 records, we implemented **Option B**: a reproducible, physics-grounded 250-shipment benchmark across 5 global trade corridors.

---

## 3. Mathematical Formulation & Leakage-Free Design

### Sequential Stage Definition
Each shipment journey $i$ traverses $K = 5$ sequential stages:
- **Stage 1 ($k=1$):** Maritime Sea Transit (Origin Departure to Destination Anchorage)
- **Stage 2 ($k=2$):** Port Anchorage, Pilotage & Berthing
- **Stage 3 ($k=3$):** Terminal Container Handling & Yard Dwell
- **Stage 4 ($k=4$):** Customs & Regulatory Clearance
- **Stage 5 ($k=5$):** Inland Drayage & Consignee Gate Delivery

### Closed Timestamp Progression
To prevent double-counting, departure delay $\Delta t_{i, \text{dep}}$ is modeled separately and not added to waterborne transit duration $d_{i,1}^{\text{actual}}$:
- **Scheduled Sequence:** $T_{i,0}^{\text{sched}}$ is scheduled departure; $T_{i,k}^{\text{sched}} = T_{i,k-1}^{\text{sched}} + d_{i,k}^{\text{sched}}$.
- **Actual Realization:** $T_{i,0}^{\text{actual}} = T_{i,0}^{\text{sched}} + \Delta t_{i, \text{dep}}$; $T_{i,k}^{\text{actual}} = T_{i,k-1}^{\text{actual}} + d_{i,k}^{\text{actual}}$.

Total final delay satisfies the exact invariant:
$$\text{final\_delay\_hours}_i = \frac{T_{i,5}^{\text{actual}} - T_{i,5}^{\text{sched}}}{3600} = \Delta t_{i, \text{dep}} + \sum_{k=1}^5 \left( d_{i,k}^{\text{actual}} - d_{i,k}^{\text{sched}} \right)$$

### Operational Prediction Cutoff ($t_{\text{pred}}$)
$$\mathbf{t_{\text{pred}} = \text{actual\_berth\_time} = T_{i,2}^{\text{actual}}}$$
*Why Future Information is Strictly Excluded:* At the moment a container ship berths, downstream events (crane moves, customs inspections, highway traffic) have not yet occurred. Including future actual timestamps or downstream durations would introduce fatal target leakage and produce an un-deployable model.

### Prediction Targets
1. **Primary Regression Target (Remaining Duration):**
   $$y_{i, \text{rem}} = d_{i,3}^{\text{actual}} + d_{i,4}^{\text{actual}} + d_{i,5}^{\text{actual}}$$
   $$\widehat{\text{ETA}}_i = T_{i,2}^{\text{actual}} + \hat{y}_{i, \text{rem}}$$
2. **Derived Final Delay:**
   $$\hat{y}_{i, \text{delay}} = \frac{T_{i,2}^{\text{actual}} + \hat{y}_{i, \text{rem}} - T_{i,5}^{\text{sched}}}{3600} \equiv D_{i,2} + \left(\hat{y}_{i, \text{rem}} - \sum_{k=3}^5 d_{i,k}^{\text{sched}}\right)$$
   *(where $D_{i,2}$ is the delay already accumulated upon berthing).*
3. **Derived Delay Risk:** $\hat{y}_{i, \text{risk}} = \mathbb{I}(\hat{y}_{i, \text{delay}} > 6.0\text{ hours})$.

### Feature Whitelist (13 Admitted Features)
- **Static Attributes:** `route_id`, `carrier_code`, `cargo_type`, `container_teu`, `transit_distance_nm`.
- **Planned Downstream Allowances:** `scheduled_terminal_hours`, `scheduled_customs_hours`, `scheduled_drayage_hours`.
- **Upstream Realizations (Known at $t_{\text{pred}}$):** `weather_severity_index`, `carried_delay_at_anchorage`, `port_congestion_index`, `carried_delay_at_berth`, `customs_inspection_flag`.

*Operational Scope of `customs_inspection_flag`:* This feature represents an administrative inspection decision assumed to be known by the time the vessel berths (such as pre-arrival electronic manifest risk-scoring transmitted by border authorities prior to docking). It explicitly represents an assumed, pre-known regulatory decision rather than an actual physical inspection outcome or downstream milestone. In a real-world deployment, this feature is valid only if regulatory agencies provide advance hold notifications before vessel berthing; real-world availability was not verified in this assessment.

All realized downstream durations ($d_3, d_4, d_5$), downstream timestamps ($T_3, T_4, T_5$), and target derivatives are strictly blacklisted from model inputs.

---

## 4. Delay Generation & Propagation Mechanics

The synthetic simulation (`src/generator.py`, random seed 42) reflects real-world operational friction:

1. **Origin Departure ($k=0$):** 45% of vessels depart on-schedule ($\Delta t_{\text{dep}} = 0$); 55% experience departure delays drawn from an Exponential distribution ($\text{scale}=4.0\text{h}$, clipped to $24.0\text{h}$).
2. **Maritime Transit ($k=1$):** Weather severity $W_i \sim \text{Normal}(2.3, 0.8)$ clipped to $[1.0, 5.0]$. Favorable weather ($W_i < 2.3$) allows slight schedule recovery; adverse weather increases transit duration ($+3.5\%$ per index unit above 2.3) plus Gaussian hydrodynamic noise $\text{Normal}(0.0, 3.5\text{h})$.
3. **Anchorage & Berthing ($k=2$):** If carried delay $D_{i,1} \le 2.0\text{ hours}$, the vessel retains its reserved pilot/tug window with zero congestion penalty. If $D_{i,1} > 2.0\text{ hours}$, it incurs an excess-delay queue penalty scaled by the port congestion index $C_i \sim \text{Beta}(2.0, 3.0)$:
   $$d_{i,2}^{\text{actual}} = \max\left(0.6 \cdot d_{i,2}^{\text{sched}}, \; d_{i,2}^{\text{sched}} + 0.35 \times \max(0.0, D_{i,1} - 2.0) \times C_i + \epsilon_{i,2}\right)$$
   where $\epsilon_{i,2} \sim \text{Normal}(0.0, 2.5\text{h})$.
4. **Terminal Handling ($k=3$):** Terminal yards absorb up to $4.0\text{h}$ of carried delay:
   $$d_{i,3}^{\text{actual}} = \max\left(0.7 \cdot d_{i,3}^{\text{sched}}, \; d_{i,3}^{\text{sched}} + 0.15 \times \max(0.0, D_{i,2}) - 4.0 + \epsilon_{i,3}\right)$$
   where $\epsilon_{i,3} \sim \text{Normal}(0.0, 3.0\text{h})$.
5. **Customs Clearance ($k=4$):** Pre-arrival manifest flag based on cargo risk. Physical inspection holds add $\Delta_{\text{inspect}} \sim \text{Uniform}(24.0, 48.0\text{h})$; automated electronic clearance clears on-schedule with administrative noise $\text{Normal}(-1.0, 2.0\text{h})$.
6. **Inland Drayage ($k=5$):** $d_{i,5}^{\text{actual}} = \max(0.6 \cdot d_{i,5}^{\text{sched}}, \; d_{i,5}^{\text{sched}} + 0.10 \cdot \max(0.0, D_{i,4}) + \epsilon_{i,5})$ with $\epsilon_{i,5} \sim \text{Normal}(0.0, 2.5\text{h})$.

---

## 5. Model Architecture & Evaluation Methodology

### Chronological Train / Test Partition
The 250 records are sorted chronologically by `scheduled_departure_time`:
- **Training Set ($N = 200$, 80%):** Earliest departures (123 on-time, 77 delayed).
- **Held-Out Test Set ($N = 50$, 20%):** Latest departures (30 on-time, 20 delayed).
- Zero temporal overlap exists, and all 5 trade corridors are represented in both sets.
- Hyperparameters were fixed a priori (`max_iter=100`, `max_depth=4`, `min_samples_leaf=10`, `learning_rate=0.08`, `random_state=42`) with **zero tuning on the test set**.

### Models Evaluated
1. **Scheduled-Duration Baseline:** $\hat{y}_{\text{base}} = d_{i,3}^{\text{sched}} + d_{i,4}^{\text{sched}} + d_{i,5}^{\text{sched}}$, which assumes downstream stages finish within their scheduled allowances.
2. **Direct HistGradientBoostingRegressor:** Pipeline predicting absolute remaining duration $y_{i,\text{rem}}$ directly using `OneHotEncoder` and `HistGradientBoostingRegressor`.
3. **Residual Learning Predictor (`ResidualETAPredictor`):** Fits on the residual target $\Delta_{i,\text{rem}} = y_{i,\text{rem}} - d_{i,\text{downstream}}^{\text{sched}}$ and reconstructs duration as $\hat{y}_{i,\text{rem}} = \max(0, d_{i,\text{downstream}}^{\text{sched}} + \hat{\Delta}_{i,\text{rem}})$, reducing route-scale target disparity across the evaluated trade corridors.
4. **Majority-Class Classifier:** Benchmark predicting the most frequent training class (On-Time = 0).

---

## 6. Experimental Results

### Performance Comparison on Held-Out Test Set ($N = 50$)

| Evaluation Metric | Scheduled Baseline | Direct GBDT | Residual GBDT | Majority-Class | Res. vs. Base |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MAE (hours)** | 10.608 hrs | 5.921 hrs | **5.414 hrs** | — | **-49.0%** |
| **RMSE (hours)** | 15.853 hrs | 7.237 hrs | **6.578 hrs** | — | **-58.5%** |
| **MAPE (%)** *(Caveat: zero-skew)* | 14.17% | 8.89% | **8.09%** | — | -42.9% |
| **Delay-Risk Precision ($\tau = 6\text{h}$)** | 0.846 | **0.882** | **0.882** | 0.000 | +4.3% |
| **Delay-Risk Recall ($\tau = 6\text{h}$)** | 0.550 | **0.750** | **0.750** | 0.000 | **+36.4%** |
| **Delay-Risk F1-Score** | 0.667 | **0.811** | **0.811** | 0.000 | **+21.6%** |

### Delay-Risk Confusion Matrices ($\tau = 6.0\text{ hours}$, Test $N = 50$)
```text
Scheduled Baseline:                     Direct & Residual GBDT:
               Predicted                               Predicted
             On-Time  Delayed                        On-Time  Delayed
True On-Time:  [ 28       2 ]          True On-Time:  [ 28       2 ]
True Delayed:  [  9      11 ]          True Delayed:  [  5      15 ]
```
- The Scheduled Baseline misclassified 9 delayed shipments as on-time (Recall: 55.0%) due to blindness to customs holds and yard friction.
- Both Direct and Residual GBDT reduced false negatives from 9 to 5, boosting recall to **75.0%** and precision to **88.2%**, achieving an F1-score of **0.811**.
- The majority-class classifier predicts On-Time for all 50 shipments, achieving 60.0% accuracy but **0.000 F1 on delayed shipments**. Both ML models significantly outperform the scheduled baseline.

---

## 7. Model Attribution & Delay Propagation Analysis

### Permutation Feature Importance (Test Set MAE Degradation)
Evaluating the increase in test MAE when each feature is randomly permuted:
1. `customs_inspection_flag`: **+8.870 hours** error (primary driver of post-berth variance).
2. `transit_distance_nm`: **+4.252 hours** error (corridor scale proxy).
3. `route_id`: **+1.638 hours** error (corridor terminal characteristics).
4. `container_teu`: **+0.053 hours** error.
5. `port_congestion_index`: **+0.048 hours** error.

*Attribution vs. Causality Disclaimer:* Permutation importance reflects the empirical sensitivity of the trained model on this dataset. It describes statistical association rather than proven physical causality.

### Stage-to-Stage Delay Transmission Slopes
Tracking cumulative delay $D_k = D_{k-1} + \delta_k$ across all 250 journeys:
- **Departure $\rightarrow$ Anchorage:** Transmission slope = **$0.555$** (maritime schedule buffers absorb origin departure delays).
- **Anchorage $\rightarrow$ Berthing:** Transmission slope = **$1.053$** (queue compounding when vessels arrive off-schedule).
- **Berthing $\rightarrow$ Terminal:** Transmission slope = **$1.008$** (container terminal turnaround compounding).
- **Terminal $\rightarrow$ Customs:** Transmission slope = **$1.051$**.
- **Customs $\rightarrow$ Drayage Delivery:** Transmission slope = **$1.079$**.

### Customs Inspection Disruption Impact
- **Inspected Containers (44 shipments):** Mean customs dwell = **52.66 hours**.
- **Auto-Cleared Containers (206 shipments):** Mean customs dwell = **16.01 hours**.
- **Net Disruption Penalty:** **+36.65 hours** of additional dwell time directly attributable to physical inspection holds.

### Unseen-Corridor Generalization (LOCO Cross-Validation)
Holding out each corridor in turn evaluates out-of-distribution route transfer across all 5 corridors:

| Held-Out Corridor | Baseline MAE | Direct GBDT MAE | Residual GBDT MAE | Residual vs. Base |
| :--- | :---: | :---: | :---: | :---: |
| **BEANR-USNYC** (3,450 nm) | 10.769 hrs | 10.766 hrs (+0.0%) | **5.715 hrs** | **+46.9%** |
| **CNSHA-USLAX** (5,700 nm) | 11.037 hrs | 8.531 hrs (+22.7%) | **4.772 hrs** | **+56.8%** |
| **INNSA-GBFXT** (6,400 nm) | 11.978 hrs | 6.756 hrs (+43.6%) | **5.513 hrs** | **+54.0%** |
| **KRPUS-JPYOK** (620 nm feeder) | 8.578 hrs | 20.779 hrs (-142.2%) | **4.459 hrs** | **+48.0%** |
| **SGSIN-NLRTM** (8,300 nm) | 9.186 hrs | 10.585 hrs (-15.2%) | **5.714 hrs** | **+37.8%** |
| **Macro Average** | 10.310 hrs | 11.483 hrs (-11.4%) | **5.235 hrs** | **+49.2%** |

*Root Cause Analysis & Residual Resolution:* Direct GBDT failed on feeder `KRPUS-JPYOK` (-142.2% deficit) because decision trees cannot extrapolate below the minimum leaf targets observed in training ($y_{\text{train}} \in [55, 115\text{h}]$ vs. feeder allowance $44\text{h}$). By reformulating the target as schedule residual $\Delta_{\text{rem}}$, targets reflect zero-centered downstream slippage ($\Delta_{\text{rem}} \approx [-4, +40\text{h}]$ in this benchmark) rather than route distance. In this experiment, Residual GBDT resolved the feeder extrapolation deficit (4.459h vs. 8.578h base) and outperformed the scheduled baseline across all 5 evaluated corridors, improving macro MAE by **+49.2%** and macro RMSE by **+58.7%** (6.778h vs. 16.394h). These gains reflect parametric behavior in this 5-corridor simulation and do not constitute universal real-world generalization guarantees.

### Customs Feature Ablation & Controlled Disruption
- **Ablation:** Omitting `customs_inspection_flag` inflates test MAE from 5.921h to 11.599h (+95.9%) and RMSE from 7.237h to 15.766h (+117.9%), dropping delay-risk F1 from 0.811 to 0.605. Without advance manifest notice, the model performs worse than the scheduled baseline (10.608h MAE).
- **Disruption Scenarios:** A +0.35 port congestion index surge shifts model forecast by +0.123h. An unannounced +24.0h physical customs hold increases model test MAE from 5.921h to 7.557h (while baseline MAE spikes to 14.928h).

### Shipment-Level Explanations & Fitted Model Sensitivity
ShipSight provides auditable shipment-level explanations (`src/shipment_explainer.py`) with strict operational separation:
- **Live Prediction Mode (`mode="live"`):** Enforces complete pre-cutoff quarantine. Reports prediction cutoff ($T_2^{\text{actual}}$), accumulated delay ($D_2$), scheduled remaining allowance, predicted remaining duration, predicted final ETA, and predicted delay relative to schedule. Downstream actual delivery timestamps, realized delays, forecast errors, and ground-truth risk are strictly suppressed.
- **Retrospective Evaluation Mode (`mode="retrospective"`):** Post-delivery audit mode comparing operational forecasts against realized delivery timestamps, actual final delay, and forecast error margins.
- **Fitted Model Scenario Sensitivity:** Evaluates the actual fitted pipeline on modified copies of admissible features (e.g., toggling `customs_inspection_flag: 0 <-> 1` shifts predicted remaining duration by +34.42h on test shipment SHP-201). These shifts are explicitly classified as statistical model sensitivities—**not physical causal attributions, not counterfactual guarantees, and not additive SHAP values**.

### Repeated-Seed Stability Evaluation (10 Deterministic Runs)
Across 10 random seeds ($N=2,500$ shipments total, 80/20 chronological split per seed):
- **Baseline:** MAE = $10.643 \pm 1.928\text{h}$, RMSE = $16.738 \pm 2.491\text{h}$, Delay-Risk F1 = $0.706 \pm 0.094$.
- **Direct GBDT:** MAE = $4.915 \pm 0.608\text{h}$ (**+53.0%** mean reduction, $\text{std}: 6.5\%$), RMSE = $6.436 \pm 0.798\text{h}$ (**+61.1%** mean reduction, $\text{std}: 5.1\%$), Delay-Risk F1 = $0.883 \pm 0.046$.
- **Residual GBDT:** MAE = **$4.678 \pm 0.405\text{h}$** (**+55.2%** mean reduction, $\text{std}: 5.8\%$), RMSE = **$6.054 \pm 0.464\text{h}$** (**+63.3%** mean reduction, $\text{std}: 4.1\%$), Delay-Risk F1 = **$0.894 \pm 0.045$**. Both ML models beat the scheduled baseline on MAE in **10 out of 10 runs (100%)**.
*(Note: Reductions are evaluated per seed as `((baseline - model) / baseline) * 100`. Mean per-seed MAE reduction is $+55.2\% \pm 5.8\%$ for Residual and $+53.0\% \pm 6.5\%$ for Direct; evaluating aggregate mean MAEs yields $56.0\%$ and $53.8\%$. Mean per-seed RMSE reduction is $+63.3\% \pm 4.1\%$ for Residual and $+61.1\% \pm 5.1\%$ for Direct; evaluating aggregate mean RMSEs yields $63.8\%$ and $61.5\%$.)*

---

## 8. Assumptions, Trade-offs & Limitations

1. **Synthetic Data Disclosure:** All 250 shipment records are simulated using parametric distributions. The benchmark is synthetic and does not represent live carrier telematics.
2. **Model Attribution $\neq$ Causality:** Permutation importance scores reflect feature sensitivity in the trained GBDT estimator, not physical causal interventions.
3. **Customs Availability Assumption:** `customs_inspection_flag` assumes pre-arrival screening notifications are accessible at berth; if unavailable in practice, downstream prediction error will increase.
4. **Sample Size ($N = 250$):** Sized for the take-home assessment. Industrial production models require multi-year datasets capturing seasonal surges and container imbalances.
5. **Single-Leg Simplification:** The system models point-to-point intermodal lanes without multi-port container restowage or feeder transshipment connections.
6. **No Production Readiness Claim:** ShipSight is an explainable proof-of-concept pipeline designed to evaluate multi-stage delay propagation without target leakage.
7. **Exploratory AIS Audit Provenance:** Preliminary telemetry audit findings from NOAA MarineCadastre and AISHub reflect exploratory scoping not bundled in or reproducible from this submitted archive; they serve solely as methodological rationale for adopting a multi-stage benchmark.
8. **Unseen Route Transfer & Schedule Reliance:** Direct duration models fail on short-sea routes due to tree leaf bounding. While residual learning eliminated the feeder deficit across the 5 synthetic test corridors (+49.2% macro MAE improvement), this is observed under our simulation equations and is not a universal generalization guarantee. It inherently relies on having reliable scheduled stage buffers at cutoff; unexpected contractual schedule shifts or corrupted schedule metadata would directly affect residual reconstruction offsets.
9. **Parametric Simulation Scope:** Multi-seed stability evaluates algorithmic consistency across pseudo-random generator draws; it does not constitute empirical significance on live telemetry.
10. **Model Sensitivity vs. Causal Attribution:** What-if scenario sensitivity reflects the local gradient of the fitted regression estimator under perturbed inputs; it does not guarantee that physical interventions on port operations or customs policy would yield identical real-world results.

---

## 9. Conclusion & Operational Recommendations for Aiqon

1. **Ingest EDI 315 / API Milestones:** Production ETA systems must combine marine AIS tracking with carrier milestone feeds containing container status codes (gate-in, customs hold, release).
2. **Dynamic Geofence Congestion:** Use real-time AIS vessel density within port anchorage polygons to dynamically update `port_congestion_index`.
3. **Two-Stage Hurdle Architecture:** Because customs inspection is the single largest disruption factor (+36.7 hours), deploy a classification model at vessel departure to forecast the probability of customs inspection, feeding this calibrated risk directly into the downstream regression model.

---
*End of Technical Report.*
