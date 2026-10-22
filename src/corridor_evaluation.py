"""Leave-One-Corridor-Out (LOCO) Generalization Evaluation for ShipSight.

Evaluates how well the remaining-duration model generalizes to completely unseen
maritime trade corridors by holding out one corridor at a time, training on the
remaining corridors, and comparing against the scheduled-duration baseline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CORRIDORS
from src.model import (
    ResidualETAPredictor,
    build_feature_matrix,
    build_pipeline,
    calculate_scheduled_remaining_duration,
    compute_metrics,
    load_dataset,
)


def split_leave_one_corridor_out(
    records: List[Dict[str, Any]],
    held_out_corridor: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records into training (all corridors except held_out_corridor)

    and test (only held_out_corridor). Ensures strict corridor separation.
    """
    if held_out_corridor not in CORRIDORS:
        raise ValueError(f"Unknown corridor '{held_out_corridor}'. Valid: {list(CORRIDORS.keys())}")

    train_records = [r for r in records if r["route_id"] != held_out_corridor]
    test_records = [r for r in records if r["route_id"] == held_out_corridor]

    if not test_records:
        raise ValueError(f"No records found for held-out corridor '{held_out_corridor}'.")
    if not train_records:
        raise ValueError(f"No training records remaining when holding out '{held_out_corridor}'.")

    # Strict isolation assertion
    train_corridors = {r["route_id"] for r in train_records}
    test_corridors = {r["route_id"] for r in test_records}
    assert held_out_corridor not in train_corridors, "Leakage: held-out corridor found in training set!"
    assert train_corridors.isdisjoint(test_corridors), "Leakage: train and test corridors overlap!"

    return train_records, test_records


def run_leave_one_corridor_out_evaluation(
    csv_path: str = os.path.join("data", "synthetic_shipments.csv"),
) -> Dict[str, Any]:
    """Executes leave-one-corridor-out cross-validation across all configured corridors,
    comparing Baseline, Direct ML, and Residual ML candidates on identical folds.
    """
    records = load_dataset(csv_path)
    corridor_keys = sorted(list(CORRIDORS.keys()))
    results_by_corridor: Dict[str, Dict[str, Any]] = {}

    total_test_samples = 0
    baseline_maes: List[float] = []
    baseline_rmses: List[float] = []
    baseline_f1s: List[float] = []

    direct_maes: List[float] = []
    direct_rmses: List[float] = []
    direct_f1s: List[float] = []

    residual_maes: List[float] = []
    residual_rmses: List[float] = []
    residual_f1s: List[float] = []

    for corridor in corridor_keys:
        train_rec, test_rec = split_leave_one_corridor_out(records, corridor)
        n_test = len(test_rec)
        total_test_samples += n_test

        X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_rec)
        X_test_dicts, y_test_rem, _, _ = build_feature_matrix(test_rec)

        y_train_sched = calculate_scheduled_remaining_duration(train_rec)
        y_test_sched = calculate_scheduled_remaining_duration(test_rec)

        X_train_df = pd.DataFrame(X_train_dicts)
        X_test_df = pd.DataFrame(X_test_dicts)

        # 1. Baseline evaluation on held-out corridor
        base_metrics = compute_metrics(
            y_test_rem,
            y_test_sched,
            test_rec,
            f"Baseline ({corridor})",
        )

        # 2. Direct ML Pipeline (predicts absolute y_rem)
        pipeline_direct = build_pipeline()
        pipeline_direct.fit(X_train_df, y_train_rem)
        y_test_pred_direct = pipeline_direct.predict(X_test_df)
        direct_metrics = compute_metrics(
            y_test_rem,
            y_test_pred_direct,
            test_rec,
            f"Direct ML ({corridor})",
        )

        # 3. Residual ML Predictor: predicts y_rem - y_sched
        residual_predictor = ResidualETAPredictor()
        residual_predictor.fit(X_train_df, y_train_rem, y_train_sched)
        y_test_pred_residual = residual_predictor.predict(X_test_df, y_test_sched)
        residual_metrics = compute_metrics(
            y_test_rem,
            y_test_pred_residual,
            test_rec,
            f"Residual ML ({corridor})",
        )

        direct_mae_improvement = (
            (base_metrics["mae_hours"] - direct_metrics["mae_hours"]) / base_metrics["mae_hours"] * 100.0
            if base_metrics["mae_hours"] > 0
            else 0.0
        )
        direct_rmse_improvement = (
            (base_metrics["rmse_hours"] - direct_metrics["rmse_hours"]) / base_metrics["rmse_hours"] * 100.0
            if base_metrics["rmse_hours"] > 0
            else 0.0
        )

        residual_mae_improvement = (
            (base_metrics["mae_hours"] - residual_metrics["mae_hours"]) / base_metrics["mae_hours"] * 100.0
            if base_metrics["mae_hours"] > 0
            else 0.0
        )
        residual_rmse_improvement = (
            (base_metrics["rmse_hours"] - residual_metrics["rmse_hours"]) / base_metrics["rmse_hours"] * 100.0
            if base_metrics["rmse_hours"] > 0
            else 0.0
        )

        results_by_corridor[corridor] = {
            "n_samples": n_test,
            "baseline_mae": base_metrics["mae_hours"],
            "baseline_rmse": base_metrics["rmse_hours"],
            "baseline_f1": base_metrics["risk_f1"],
            "direct_mae": direct_metrics["mae_hours"],
            "direct_rmse": direct_metrics["rmse_hours"],
            "direct_f1": direct_metrics["risk_f1"],
            "direct_mae_improvement_pct": round(direct_mae_improvement, 1),
            "direct_rmse_improvement_pct": round(direct_rmse_improvement, 1),
            "residual_mae": residual_metrics["mae_hours"],
            "residual_rmse": residual_metrics["rmse_hours"],
            "residual_f1": residual_metrics["risk_f1"],
            "residual_mae_improvement_pct": round(residual_mae_improvement, 1),
            "residual_rmse_improvement_pct": round(residual_rmse_improvement, 1),
            # Backwards-compatible aliases pointing to residual candidate:
            "model_mae": residual_metrics["mae_hours"],
            "model_rmse": residual_metrics["rmse_hours"],
            "mae_improvement_pct": round(residual_mae_improvement, 1),
            "rmse_improvement_pct": round(residual_rmse_improvement, 1),
        }

        baseline_maes.append(base_metrics["mae_hours"])
        baseline_rmses.append(base_metrics["rmse_hours"])
        baseline_f1s.append(base_metrics["risk_f1"])

        direct_maes.append(direct_metrics["mae_hours"])
        direct_rmses.append(direct_metrics["rmse_hours"])
        direct_f1s.append(direct_metrics["risk_f1"])

        residual_maes.append(residual_metrics["mae_hours"])
        residual_rmses.append(residual_metrics["rmse_hours"])
        residual_f1s.append(residual_metrics["risk_f1"])

    # Macro averages across corridors
    macro_summary = {
        "mean_baseline_mae": round(float(np.mean(baseline_maes)), 3),
        "mean_baseline_rmse": round(float(np.mean(baseline_rmses)), 3),
        "mean_baseline_f1": round(float(np.mean(baseline_f1s)), 3),
        "mean_direct_mae": round(float(np.mean(direct_maes)), 3),
        "mean_direct_rmse": round(float(np.mean(direct_rmses)), 3),
        "mean_direct_f1": round(float(np.mean(direct_f1s)), 3),
        "mean_residual_mae": round(float(np.mean(residual_maes)), 3),
        "mean_residual_rmse": round(float(np.mean(residual_rmses)), 3),
        "mean_residual_f1": round(float(np.mean(residual_f1s)), 3),
        # Backwards-compatible aliases:
        "mean_model_mae": round(float(np.mean(residual_maes)), 3),
        "mean_model_rmse": round(float(np.mean(residual_rmses)), 3),
        "macro_mae_improvement_pct": round(
            float((np.mean(baseline_maes) - np.mean(residual_maes)) / np.mean(baseline_maes) * 100.0), 1
        ),
        "macro_rmse_improvement_pct": round(
            float((np.mean(baseline_rmses) - np.mean(residual_rmses)) / np.mean(baseline_rmses) * 100.0), 1
        ),
        "direct_macro_mae_improvement_pct": round(
            float((np.mean(baseline_maes) - np.mean(direct_maes)) / np.mean(baseline_maes) * 100.0), 1
        ),
        "direct_macro_rmse_improvement_pct": round(
            float((np.mean(baseline_rmses) - np.mean(direct_rmses)) / np.mean(baseline_rmses) * 100.0), 1
        ),
        "total_evaluated_records": total_test_samples,
    }

    return {
        "by_corridor": results_by_corridor,
        "macro_summary": macro_summary,
    }


if __name__ == "__main__":
    res = run_leave_one_corridor_out_evaluation()
    print("=" * 84)
    print("LEAVE-ONE-CORRIDOR-OUT (LOCO) GENERALIZATION EVALUATION")
    print("=" * 84)
    print(f"{'Corridor':<12} | {'Baseline MAE':<12} | {'Direct ML MAE':<15} | {'Residual ML MAE':<15} | {'Residual Impv'}")
    print("-" * 84)
    for corr, m in res["by_corridor"].items():
        print(
            f"{corr:<12} | {m['baseline_mae']:>6.3f} hrs    | {m['direct_mae']:>6.3f} hrs ({m['direct_mae_improvement_pct']:>+5.1f}%) | "
            f"{m['residual_mae']:>6.3f} hrs ({m['residual_mae_improvement_pct']:>+5.1f}%) | {m['residual_mae_improvement_pct']:>+5.1f}%"
        )
    print("-" * 84)
    ms = res["macro_summary"]
    print(
        f"{'Macro Mean':<12} | {ms['mean_baseline_mae']:>6.3f} hrs    | {ms['mean_direct_mae']:>6.3f} hrs ({ms['direct_macro_mae_improvement_pct']:>+5.1f}%) | "
        f"{ms['mean_residual_mae']:>6.3f} hrs ({ms['macro_mae_improvement_pct']:>+5.1f}%) | {ms['macro_mae_improvement_pct']:>+5.1f}%"
    )
    print("=" * 84)
