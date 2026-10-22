"""Repeated-Seed Evaluation for ShipSight ETA Prediction Pipeline.

Evaluates the stability and consistency of the HistGradientBoostingRegressor
against the scheduled-duration baseline across multiple deterministic random seeds,
generating a fresh synthetic shipment dataset for each seed while strictly preserving
chronological train/test splitting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import NUM_SHIPMENTS, RANDOM_SEED, TRAIN_RATIO
from src.generator import generate_synthetic_shipments
from src.model import (
    build_feature_matrix,
    build_pipeline,
    compute_metrics,
    split_chronological,
)

DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def evaluate_single_seed(
    seed: int,
    num_shipments: int = NUM_SHIPMENTS,
    train_ratio: float = TRAIN_RATIO,
) -> Dict[str, Any]:
    """Generates a dataset with the specified seed and evaluates baseline vs. ML model.

    Preserves chronological train/test splitting. Fits preprocessor and regressor
    strictly on the training partition.
    """
    records = generate_synthetic_shipments(num_shipments=num_shipments, seed=seed)
    train_records, test_records = split_chronological(records, train_ratio=train_ratio)

    X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_records)
    X_test_dicts, y_test_rem, _, _ = build_feature_matrix(test_records)

    # 1. Scheduled Duration Baseline on Test Partition
    y_test_pred_base = np.array(
        [
            r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"]
            for r in test_records
        ],
        dtype=np.float64,
    )
    baseline_metrics = compute_metrics(
        y_test_rem,
        y_test_pred_base,
        test_records,
        "Scheduled Duration Baseline",
    )

    # 2. HistGradientBoostingRegressor Pipeline (fit strictly on train)
    pipeline = build_pipeline()
    X_train_df = pd.DataFrame(X_train_dicts)
    X_test_df = pd.DataFrame(X_test_dicts)
    pipeline.fit(X_train_df, y_train_rem)

    y_test_pred_ml = pipeline.predict(X_test_df)
    ml_metrics = compute_metrics(
        y_test_rem,
        y_test_pred_ml,
        test_records,
        "HistGradientBoosting",
    )

    mae_improvement_pct = (
        (baseline_metrics["mae_hours"] - ml_metrics["mae_hours"]) / baseline_metrics["mae_hours"] * 100.0
        if baseline_metrics["mae_hours"] > 0
        else 0.0
    )
    rmse_improvement_pct = (
        (baseline_metrics["rmse_hours"] - ml_metrics["rmse_hours"]) / baseline_metrics["rmse_hours"] * 100.0
        if baseline_metrics["rmse_hours"] > 0
        else 0.0
    )

    model_beats_baseline_mae = bool(ml_metrics["mae_hours"] < baseline_metrics["mae_hours"])

    return {
        "seed": seed,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "baseline_mae": baseline_metrics["mae_hours"],
        "baseline_rmse": baseline_metrics["rmse_hours"],
        "baseline_f1": baseline_metrics["risk_f1"],
        "ml_mae": ml_metrics["mae_hours"],
        "ml_rmse": ml_metrics["rmse_hours"],
        "ml_precision": ml_metrics["risk_precision"],
        "ml_recall": ml_metrics["risk_recall"],
        "ml_f1": ml_metrics["risk_f1"],
        "mae_improvement_pct": round(mae_improvement_pct, 2),
        "rmse_improvement_pct": round(rmse_improvement_pct, 2),
        "model_beats_baseline_mae": model_beats_baseline_mae,
    }


def run_repeated_seed_evaluation(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_shipments: int = NUM_SHIPMENTS,
    train_ratio: float = TRAIN_RATIO,
) -> Dict[str, Any]:
    """Runs evaluation across multiple seeds and compiles summary statistics (mean, std, win count)."""
    run_results = [
        evaluate_single_seed(seed=s, num_shipments=num_shipments, train_ratio=train_ratio)
        for s in seeds
    ]

    base_maes = [r["baseline_mae"] for r in run_results]
    base_rmses = [r["baseline_rmse"] for r in run_results]
    base_f1s = [r["baseline_f1"] for r in run_results]

    ml_maes = [r["ml_mae"] for r in run_results]
    ml_rmses = [r["ml_rmse"] for r in run_results]
    ml_precs = [r["ml_precision"] for r in run_results]
    ml_recs = [r["ml_recall"] for r in run_results]
    ml_f1s = [r["ml_f1"] for r in run_results]
    mae_impvs = [r["mae_improvement_pct"] for r in run_results]
    rmse_impvs = [r["rmse_improvement_pct"] for r in run_results]

    wins = sum(1 for r in run_results if r["model_beats_baseline_mae"])
    total_runs = len(run_results)

    summary = {
        "num_runs": total_runs,
        "seeds_evaluated": list(seeds),
        "model_beat_baseline_mae_count": wins,
        "win_rate_pct": round((wins / total_runs) * 100.0, 1) if total_runs > 0 else 0.0,
        "baseline_mae": {
            "mean": round(float(np.mean(base_maes)), 3),
            "std": round(float(np.std(base_maes)), 3),
            "min": round(float(np.min(base_maes)), 3),
            "max": round(float(np.max(base_maes)), 3),
        },
        "baseline_rmse": {
            "mean": round(float(np.mean(base_rmses)), 3),
            "std": round(float(np.std(base_rmses)), 3),
        },
        "baseline_f1": {
            "mean": round(float(np.mean(base_f1s)), 3),
            "std": round(float(np.std(base_f1s)), 3),
        },
        "ml_mae": {
            "mean": round(float(np.mean(ml_maes)), 3),
            "std": round(float(np.std(ml_maes)), 3),
            "min": round(float(np.min(ml_maes)), 3),
            "max": round(float(np.max(ml_maes)), 3),
        },
        "ml_rmse": {
            "mean": round(float(np.mean(ml_rmses)), 3),
            "std": round(float(np.std(ml_rmses)), 3),
        },
        "ml_precision": {
            "mean": round(float(np.mean(ml_precs)), 3),
            "std": round(float(np.std(ml_precs)), 3),
        },
        "ml_recall": {
            "mean": round(float(np.mean(ml_recs)), 3),
            "std": round(float(np.std(ml_recs)), 3),
        },
        "ml_f1": {
            "mean": round(float(np.mean(ml_f1s)), 3),
            "std": round(float(np.std(ml_f1s)), 3),
        },
        "mae_improvement_pct": {
            "mean": round(float(np.mean(mae_impvs)), 2),
            "std": round(float(np.std(mae_impvs)), 2),
        },
        "rmse_improvement_pct": {
            "mean": round(float(np.mean(rmse_impvs)), 2),
            "std": round(float(np.std(rmse_impvs)), 2),
        },
        "methodology_disclaimer": (
            "Disclaimer: Results reflect repeated synthetic simulations across deterministic random seeds. "
            "They quantify parametric simulation stability under the generative model, and must not be "
            "interpreted as empirical statistical significance or real-world generalization on live maritime telemetry."
        ),
    }

    return {
        "runs": run_results,
        "summary": summary,
    }


def format_repeated_seed_summary(results: Dict[str, Any]) -> str:
    """Formats the repeated seed results into a clean text summary table."""
    s = results["summary"]
    lines = [
        "=" * 70,
        f" REPEATED-SEED EVALUATION SUMMARY ({s['num_runs']} Deterministic Runs)",
        "=" * 70,
        f"Seeds Evaluated: {s['seeds_evaluated']}",
        f"Runs ML Beat Baseline on MAE: {s['model_beat_baseline_mae_count']} / {s['num_runs']} ({s['win_rate_pct']}%)",
        "-" * 70,
        f"{'Metric':<28} | {'Baseline (Mean +/- Std)':<22} | {'ML Model (Mean +/- Std)':<22}",
        "-" * 70,
        f"{'MAE (hours)':<28} | {s['baseline_mae']['mean']:>6.3f} +/- {s['baseline_mae']['std']:<11.3f} | {s['ml_mae']['mean']:>6.3f} +/- {s['ml_mae']['std']:<11.3f}",
        f"{'RMSE (hours)':<28} | {s['baseline_rmse']['mean']:>6.3f} +/- {s['baseline_rmse']['std']:<11.3f} | {s['ml_rmse']['mean']:>6.3f} +/- {s['ml_rmse']['std']:<11.3f}",
        f"{'Delay Risk F1':<28} | {s['baseline_f1']['mean']:>6.3f} +/- {s['baseline_f1']['std']:<11.3f} | {s['ml_f1']['mean']:>6.3f} +/- {s['ml_f1']['std']:<11.3f}",
        f"{'Delay Risk Precision':<28} | {'-':<22} | {s['ml_precision']['mean']:>6.3f} +/- {s['ml_precision']['std']:<11.3f}",
        f"{'Delay Risk Recall':<28} | {'-':<22} | {s['ml_recall']['mean']:>6.3f} +/- {s['ml_recall']['std']:<11.3f}",
        "-" * 70,
        f"Mean MAE Improvement : -{s['mae_improvement_pct']['mean']:.1f}% (std: {s['mae_improvement_pct']['std']:.1f}%)",
        f"Mean RMSE Improvement: -{s['rmse_improvement_pct']['mean']:.1f}% (std: {s['rmse_improvement_pct']['std']:.1f}%)",
        "-" * 70,
        f"Note: {s['methodology_disclaimer']}",
        "=" * 70,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_repeated_seed_evaluation()
    print(format_repeated_seed_summary(res))
