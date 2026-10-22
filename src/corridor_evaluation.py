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
    build_feature_matrix,
    build_pipeline,
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
    """Executes leave-one-corridor-out cross-validation across all configured corridors."""
    records = load_dataset(csv_path)
    corridor_keys = sorted(list(CORRIDORS.keys()))
    results_by_corridor: Dict[str, Dict[str, Any]] = {}

    total_test_samples = 0
    baseline_maes: List[float] = []
    baseline_rmses: List[float] = []
    model_maes: List[float] = []
    model_rmses: List[float] = []

    for corridor in corridor_keys:
        train_rec, test_rec = split_leave_one_corridor_out(records, corridor)
        n_test = len(test_rec)
        total_test_samples += n_test

        X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_rec)
        X_test_dicts, y_test_rem, _, _ = build_feature_matrix(test_rec)

        # Baseline evaluation on held-out corridor
        y_test_pred_base = np.array(
            [
                r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"]
                for r in test_rec
            ],
            dtype=np.float64,
        )
        base_metrics = compute_metrics(
            y_test_rem,
            y_test_pred_base,
            test_rec,
            f"Baseline ({corridor})",
        )

        # Train pipeline on remaining corridors
        pipeline = build_pipeline()
        X_train_df = pd.DataFrame(X_train_dicts)
        X_test_df = pd.DataFrame(X_test_dicts)
        pipeline.fit(X_train_df, y_train_rem)

        y_test_pred_model = pipeline.predict(X_test_df)
        model_metrics = compute_metrics(
            y_test_rem,
            y_test_pred_model,
            test_rec,
            f"HistGradientBoosting ({corridor})",
        )

        mae_improvement = (
            (base_metrics["mae_hours"] - model_metrics["mae_hours"]) / base_metrics["mae_hours"] * 100.0
            if base_metrics["mae_hours"] > 0
            else 0.0
        )
        rmse_improvement = (
            (base_metrics["rmse_hours"] - model_metrics["rmse_hours"]) / base_metrics["rmse_hours"] * 100.0
            if base_metrics["rmse_hours"] > 0
            else 0.0
        )

        results_by_corridor[corridor] = {
            "n_samples": n_test,
            "baseline_mae": base_metrics["mae_hours"],
            "baseline_rmse": base_metrics["rmse_hours"],
            "model_mae": model_metrics["mae_hours"],
            "model_rmse": model_metrics["rmse_hours"],
            "mae_improvement_pct": round(mae_improvement, 1),
            "rmse_improvement_pct": round(rmse_improvement, 1),
        }

        baseline_maes.append(base_metrics["mae_hours"])
        baseline_rmses.append(base_metrics["rmse_hours"])
        model_maes.append(model_metrics["mae_hours"])
        model_rmses.append(model_metrics["rmse_hours"])

    # Macro averages across corridors
    macro_summary = {
        "mean_baseline_mae": round(float(np.mean(baseline_maes)), 3),
        "mean_baseline_rmse": round(float(np.mean(baseline_rmses)), 3),
        "mean_model_mae": round(float(np.mean(model_maes)), 3),
        "mean_model_rmse": round(float(np.mean(model_rmses)), 3),
        "macro_mae_improvement_pct": round(
            float((np.mean(baseline_maes) - np.mean(model_maes)) / np.mean(baseline_maes) * 100.0), 1
        ),
        "macro_rmse_improvement_pct": round(
            float((np.mean(baseline_rmses) - np.mean(model_rmses)) / np.mean(baseline_rmses) * 100.0), 1
        ),
        "total_evaluated_records": total_test_samples,
    }

    return {
        "by_corridor": results_by_corridor,
        "macro_summary": macro_summary,
    }


if __name__ == "__main__":
    res = run_leave_one_corridor_out_evaluation()
    print("=" * 70)
    print("LEAVE-ONE-CORRIDOR-OUT (LOCO) GENERALIZATION EVALUATION")
    print("=" * 70)
    for corr, m in res["by_corridor"].items():
        print(
            f"Corridor {corr:<12} (N={m['n_samples']:<2}): "
            f"Baseline MAE={m['baseline_mae']:>6.3f}h -> Model MAE={m['model_mae']:>6.3f}h "
            f"({m['mae_improvement_pct']:>+5.1f}%) | "
            f"RMSE: {m['baseline_rmse']:>6.3f}h -> {m['model_rmse']:>6.3f}h"
        )
    print("-" * 70)
    ms = res["macro_summary"]
    print(
        f"Macro Average: Baseline MAE={ms['mean_baseline_mae']:.3f}h -> "
        f"Model MAE={ms['mean_model_mae']:.3f}h ({ms['macro_mae_improvement_pct']:+.1f}%) | "
        f"RMSE={ms['mean_baseline_rmse']:.3f}h -> {ms['mean_model_rmse']:.3f}h ({ms['macro_rmse_improvement_pct']:+.1f}%)"
    )
