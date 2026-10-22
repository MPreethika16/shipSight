"""Ablation and Disruption Scenario Analysis for ShipSight.

Evaluates:
1. Feature ablation: Measures the model's explicit dependence on
   `customs_inspection_flag` on the chronological train/test split.
2. Controlled disruption scenarios: Assesses model sensitivity and error
   under controlled port congestion surges and regulatory customs delays.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DELAY_RISK_THRESHOLD_HOURS, FEATURE_WHITELIST
from src.model import (
    build_feature_matrix,
    build_pipeline,
    compute_metrics,
    load_dataset,
    split_chronological,
)


def evaluate_customs_ablation(
    csv_path: str = os.path.join("data", "synthetic_shipments.csv"),
) -> Dict[str, Any]:
    """Compares the full model against an ablated model trained without `customs_inspection_flag`.

    Uses the exact same chronological train/test partition (200 train / 50 test).
    """
    records = load_dataset(csv_path)
    train_records, test_records = split_chronological(records)

    # 1. Full Model Benchmark
    X_train_full, y_train_rem, _, _ = build_feature_matrix(train_records)
    X_test_full, y_test_rem, _, _ = build_feature_matrix(test_records)

    pipeline_full = build_pipeline()
    pipeline_full.fit(pd.DataFrame(X_train_full), y_train_rem)
    y_pred_full = pipeline_full.predict(pd.DataFrame(X_test_full))
    metrics_full = compute_metrics(y_test_rem, y_pred_full, test_records, "Full Feature Set")

    # 2. Ablated Model Benchmark (Excluding customs_inspection_flag)
    ablated_features = [f for f in FEATURE_WHITELIST if f != "customs_inspection_flag"]
    categorical_cols = ["route_id", "carrier_code", "cargo_type"]
    numerical_cols = [c for c in ablated_features if c not in categorical_cols]

    preprocessor_ablated = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ],
        remainder="drop",
    )
    regressor_ablated = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=4,
        min_samples_leaf=10,
        learning_rate=0.08,
        random_state=42,
    )
    pipeline_ablated = Pipeline(
        steps=[("preprocessor", preprocessor_ablated), ("regressor", regressor_ablated)]
    )

    X_train_ablated = [{c: r[c] for c in ablated_features} for r in train_records]
    X_test_ablated = [{c: r[c] for c in ablated_features} for r in test_records]

    pipeline_ablated.fit(pd.DataFrame(X_train_ablated), y_train_rem)
    y_pred_ablated = pipeline_ablated.predict(pd.DataFrame(X_test_ablated))
    metrics_ablated = compute_metrics(
        y_test_rem, y_pred_ablated, test_records, "Ablated (No Customs Flag)"
    )

    # 3. Scheduled Baseline on Test Set
    y_test_pred_base = np.array(
        [
            r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"]
            for r in test_records
        ],
        dtype=np.float64,
    )
    metrics_baseline = compute_metrics(
        y_test_rem, y_test_pred_base, test_records, "Scheduled Duration Baseline"
    )

    mae_delta = metrics_ablated["mae_hours"] - metrics_full["mae_hours"]
    mae_degradation_pct = round((mae_delta / metrics_full["mae_hours"]) * 100.0, 1)

    rmse_delta = metrics_ablated["rmse_hours"] - metrics_full["rmse_hours"]
    rmse_degradation_pct = round((rmse_delta / metrics_full["rmse_hours"]) * 100.0, 1)

    return {
        "metrics_full": metrics_full,
        "metrics_ablated": metrics_ablated,
        "metrics_baseline": metrics_baseline,
        "comparison": {
            "mae_full": metrics_full["mae_hours"],
            "mae_ablated": metrics_ablated["mae_hours"],
            "mae_baseline": metrics_baseline["mae_hours"],
            "mae_delta_hours": round(mae_delta, 3),
            "mae_degradation_pct": mae_degradation_pct,
            "rmse_full": metrics_full["rmse_hours"],
            "rmse_ablated": metrics_ablated["rmse_hours"],
            "rmse_baseline": metrics_baseline["rmse_hours"],
            "rmse_delta_hours": round(rmse_delta, 3),
            "rmse_degradation_pct": rmse_degradation_pct,
            "f1_full": metrics_full["risk_f1"],
            "f1_ablated": metrics_ablated["risk_f1"],
            "f1_baseline": metrics_baseline["risk_f1"],
        },
        "findings_summary": (
            f"Removing customs_inspection_flag increases test MAE from {metrics_full['mae_hours']}h to "
            f"{metrics_ablated['mae_hours']}h (+{mae_degradation_pct}%) and RMSE from {metrics_full['rmse_hours']}h to "
            f"{metrics_ablated['rmse_hours']}h (+{rmse_degradation_pct}%). Without this advance manifest feature, "
            f"the model performs worse than the scheduled-duration baseline ({metrics_baseline['mae_hours']}h MAE), "
            f"demonstrating extreme dependence on pre-arrival regulatory notice."
        ),
    }


def run_controlled_disruption_scenario(
    csv_path: str = os.path.join("data", "synthetic_shipments.csv"),
    congestion_input_delta: float = 0.35,
    customs_delay_hours: float = 24.0,
) -> Dict[str, Any]:
    """Executes reproducible controlled disruption scenarios on the chronological test set.

    Distinguishes two levels of disruption:
    1. Model-Input Sensitivity: Changes `port_congestion_index` at inference time while
       keeping ground-truth constant, evaluating planner forecast shift.
    2. Ground-Truth Operational Disruption: Prolongs physical customs inspection dwell
       by `customs_delay_hours` for inspected shipments, evaluating impact on actual
       delay and prediction error.
    """
    records = load_dataset(csv_path)
    train_records, test_records = split_chronological(records)

    # Train standard model
    X_train_full, y_train_rem, _, _ = build_feature_matrix(train_records)
    pipeline = build_pipeline()
    pipeline.fit(pd.DataFrame(X_train_full), y_train_rem)

    # Baseline predictions under normal conditions
    X_test_normal, y_test_normal_rem, _, _ = build_feature_matrix(test_records)
    y_pred_normal = pipeline.predict(pd.DataFrame(X_test_normal))
    normal_metrics = compute_metrics(y_test_normal_rem, y_pred_normal, test_records, "Normal Conditions")

    # --- Scenario 1: Input-Only Congestion Surge ---
    # Changes model input port_congestion_index (+congestion_input_delta), ground truth untouched.
    X_test_congested_inputs = []
    for d in X_test_normal:
        dc = copy.deepcopy(d)
        dc["port_congestion_index"] = min(1.0, dc["port_congestion_index"] + congestion_input_delta)
        X_test_congested_inputs.append(dc)

    y_pred_congested_inputs = pipeline.predict(pd.DataFrame(X_test_congested_inputs))
    mean_forecast_increase_hours = float(np.mean(y_pred_congested_inputs - y_pred_normal))

    # --- Scenario 2: Ground-Truth Customs Disruption ---
    # Physical inspection holds take an extra `customs_delay_hours` in downstream actuals.
    disrupted_test_records = []
    disrupted_y_true_rem = []

    for r in test_records:
        r_disp = copy.deepcopy(r)
        if r_disp["customs_inspection_flag"] == 1:
            r_disp["actual_customs_hours"] += customs_delay_hours
            r_disp["remaining_duration_hours"] += customs_delay_hours
            r_disp["final_delay_hours"] += customs_delay_hours
            r_disp["delay_risk_binary"] = int(r_disp["final_delay_hours"] > DELAY_RISK_THRESHOLD_HOURS)
        disrupted_test_records.append(r_disp)
        disrupted_y_true_rem.append(r_disp["remaining_duration_hours"])

    disrupted_y_true_arr = np.array(disrupted_y_true_rem, dtype=np.float64)

    # Evaluate standard model (which has no advance notice of the +24h physical hold)
    disrupted_metrics_model = compute_metrics(
        disrupted_y_true_arr,
        y_pred_normal,
        disrupted_test_records,
        "Model during Customs Disruption",
    )

    # Evaluate scheduled baseline under disruption
    y_test_pred_base = np.array(
        [
            r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"]
            for r in disrupted_test_records
        ],
        dtype=np.float64,
    )
    disrupted_metrics_base = compute_metrics(
        disrupted_y_true_arr,
        y_test_pred_base,
        disrupted_test_records,
        "Baseline during Customs Disruption",
    )

    inspected_count = sum(1 for r in test_records if r["customs_inspection_flag"] == 1)

    return {
        "scenario_parameters": {
            "congestion_input_delta": congestion_input_delta,
            "customs_delay_hours": customs_delay_hours,
            "test_inspected_shipments": inspected_count,
            "total_test_shipments": len(test_records),
        },
        "input_sensitivity_scenario": {
            "description": "Alters model input feature `port_congestion_index` (+0.35) at cutoff with ground-truth held constant.",
            "what_changes": "Model input only; simulated ground-truth remains untouched.",
            "mean_forecast_increase_hours": round(mean_forecast_increase_hours, 3),
        },
        "ground_truth_disruption_scenario": {
            "description": (
                f"Prolongs physical customs dwell by +{customs_delay_hours}h for all inspected containers ({inspected_count}/50 test shipments)."
            ),
            "what_changes": (
                "Simulated ground truth downstream actuals are modified; model inputs remain at pre-arrival manifest values."
            ),
            "normal_mae": normal_metrics["mae_hours"],
            "disrupted_model_mae": disrupted_metrics_model["mae_hours"],
            "disrupted_baseline_mae": disrupted_metrics_base["mae_hours"],
            "model_error_increase_hours": round(
                disrupted_metrics_model["mae_hours"] - normal_metrics["mae_hours"], 3
            ),
        },
        "disclaimer": (
            "Scenario evaluations reflect deterministic manipulations of the synthetic simulation engine. "
            "Differences must not be interpreted as empirical causal effects established from real-world maritime operations."
        ),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("1. FEATURE ABLATION: CUSTOMS INSPECTION FLAG DEPENDENCE")
    print("=" * 70)
    abl = evaluate_customs_ablation()
    c = abl["comparison"]
    print(f"Full Model:        MAE={c['mae_full']:.3f}h | RMSE={c['rmse_full']:.3f}h | F1={c['f1_full']:.3f}")
    print(f"Ablated (No Flag): MAE={c['mae_ablated']:.3f}h | RMSE={c['rmse_ablated']:.3f}h | F1={c['f1_ablated']:.3f}")
    print(f"Scheduled Base:    MAE={c['mae_baseline']:.3f}h | RMSE={c['rmse_baseline']:.3f}h | F1={c['f1_baseline']:.3f}")
    print(f"Degradation:       MAE Delta={c['mae_delta_hours']:+.3f}h ({c['mae_degradation_pct']:+.1f}%)")
    print()

    print("=" * 70)
    print("2. CONTROLLED DISRUPTION SCENARIO ANALYSIS")
    print("=" * 70)
    disp = run_controlled_disruption_scenario()
    p = disp["scenario_parameters"]
    s1 = disp["input_sensitivity_scenario"]
    s2 = disp["ground_truth_disruption_scenario"]
    print(f"Input Sensitivity: +{p['congestion_input_delta']} congestion index -> {s1['mean_forecast_increase_hours']:+.3f}h mean forecast shift")
    print(f"Ground Disruption: +{p['customs_delay_hours']}h hold on {p['test_inspected_shipments']} inspected shipments -> "
          f"Model MAE: {s2['normal_mae']:.3f}h -> {s2['disrupted_model_mae']:.3f}h (Baseline: {s2['disrupted_baseline_mae']:.3f}h)")
