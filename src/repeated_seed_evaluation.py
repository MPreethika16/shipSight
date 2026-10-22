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
    ResidualETAPredictor,
    build_feature_matrix,
    build_pipeline,
    calculate_scheduled_remaining_duration,
    compute_metrics,
    split_chronological,
)

DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def evaluate_single_seed(
    seed: int,
    num_shipments: int = NUM_SHIPMENTS,
    train_ratio: float = TRAIN_RATIO,
) -> Dict[str, Any]:
    """Generates a dataset with the specified seed and evaluates baseline, direct ML,
    and residual ML models on identical chronological partitions.
    """
    records = generate_synthetic_shipments(num_shipments=num_shipments, seed=seed)
    train_records, test_records = split_chronological(records, train_ratio=train_ratio)

    X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_records)
    X_test_dicts, y_test_rem, _, _ = build_feature_matrix(test_records)

    y_train_sched = calculate_scheduled_remaining_duration(train_records)
    y_test_sched = calculate_scheduled_remaining_duration(test_records)

    X_train_df = pd.DataFrame(X_train_dicts)
    X_test_df = pd.DataFrame(X_test_dicts)

    # 1. Scheduled Duration Baseline on Test Partition
    baseline_metrics = compute_metrics(
        y_test_rem,
        y_test_sched,
        test_records,
        "Scheduled Duration Baseline",
    )

    # 2. HistGradientBoostingRegressor Pipeline (Direct Absolute Target)
    pipeline_direct = build_pipeline()
    pipeline_direct.fit(X_train_df, y_train_rem)
    y_test_pred_direct = pipeline_direct.predict(X_test_df)
    direct_metrics = compute_metrics(
        y_test_rem,
        y_test_pred_direct,
        test_records,
        "HistGradientBoosting (Direct)",
    )

    # 3. Residual Learning Predictor: y_residual = y_rem - y_sched
    residual_predictor = ResidualETAPredictor()
    residual_predictor.fit(X_train_df, y_train_rem, y_train_sched)
    y_test_pred_residual = residual_predictor.predict(X_test_df, y_test_sched)
    residual_metrics = compute_metrics(
        y_test_rem,
        y_test_pred_residual,
        test_records,
        "HistGradientBoosting (Residual)",
    )

    direct_mae_improvement_pct = (
        (baseline_metrics["mae_hours"] - direct_metrics["mae_hours"]) / baseline_metrics["mae_hours"] * 100.0
        if baseline_metrics["mae_hours"] > 0
        else 0.0
    )
    direct_rmse_improvement_pct = (
        (baseline_metrics["rmse_hours"] - direct_metrics["rmse_hours"]) / baseline_metrics["rmse_hours"] * 100.0
        if baseline_metrics["rmse_hours"] > 0
        else 0.0
    )
    residual_mae_improvement_pct = (
        (baseline_metrics["mae_hours"] - residual_metrics["mae_hours"]) / baseline_metrics["mae_hours"] * 100.0
        if baseline_metrics["mae_hours"] > 0
        else 0.0
    )
    residual_rmse_improvement_pct = (
        (baseline_metrics["rmse_hours"] - residual_metrics["rmse_hours"]) / baseline_metrics["rmse_hours"] * 100.0
        if baseline_metrics["rmse_hours"] > 0
        else 0.0
    )

    return {
        "seed": seed,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "baseline_mae": baseline_metrics["mae_hours"],
        "baseline_rmse": baseline_metrics["rmse_hours"],
        "baseline_f1": baseline_metrics["risk_f1"],
        # Direct Model Metrics (preserved for compatibility):
        "ml_mae": direct_metrics["mae_hours"],
        "ml_rmse": direct_metrics["rmse_hours"],
        "ml_precision": direct_metrics["risk_precision"],
        "ml_recall": direct_metrics["risk_recall"],
        "ml_f1": direct_metrics["risk_f1"],
        "mae_improvement_pct": round(direct_mae_improvement_pct, 2),
        "rmse_improvement_pct": round(direct_rmse_improvement_pct, 2),
        "direct_mae": direct_metrics["mae_hours"],
        "direct_rmse": direct_metrics["rmse_hours"],
        "direct_f1": direct_metrics["risk_f1"],
        "direct_mae_improvement_pct": round(direct_mae_improvement_pct, 2),
        "direct_rmse_improvement_pct": round(direct_rmse_improvement_pct, 2),
        # Residual Model Metrics:
        "residual_mae": residual_metrics["mae_hours"],
        "residual_rmse": residual_metrics["rmse_hours"],
        "residual_precision": residual_metrics["risk_precision"],
        "residual_recall": residual_metrics["risk_recall"],
        "residual_f1": residual_metrics["risk_f1"],
        "residual_mae_improvement_pct": round(residual_mae_improvement_pct, 2),
        "residual_rmse_improvement_pct": round(residual_rmse_improvement_pct, 2),
        "model_beats_baseline_mae": bool(direct_metrics["mae_hours"] < baseline_metrics["mae_hours"]),
        "residual_beats_baseline_mae": bool(residual_metrics["mae_hours"] < baseline_metrics["mae_hours"]),
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

    direct_maes = [r["direct_mae"] for r in run_results]
    direct_rmses = [r["direct_rmse"] for r in run_results]
    direct_f1s = [r["direct_f1"] for r in run_results]

    residual_maes = [r["residual_mae"] for r in run_results]
    residual_rmses = [r["residual_rmse"] for r in run_results]
    residual_precs = [r["residual_precision"] for r in run_results]
    residual_recs = [r["residual_recall"] for r in run_results]
    residual_f1s = [r["residual_f1"] for r in run_results]
    residual_mae_impvs = [r["residual_mae_improvement_pct"] for r in run_results]
    residual_rmse_impvs = [r["residual_rmse_improvement_pct"] for r in run_results]
    direct_mae_impvs = [r["direct_mae_improvement_pct"] for r in run_results]
    direct_rmse_impvs = [r["direct_rmse_improvement_pct"] for r in run_results]

    direct_wins = sum(1 for r in run_results if r["model_beats_baseline_mae"])
    residual_wins = sum(1 for r in run_results if r["residual_beats_baseline_mae"])
    total_runs = len(run_results)

    summary = {
        "num_runs": total_runs,
        "seeds_evaluated": list(seeds),
        "model_beat_baseline_mae_count": direct_wins,
        "win_rate_pct": round((direct_wins / total_runs) * 100.0, 1) if total_runs > 0 else 0.0,
        "residual_beat_baseline_mae_count": residual_wins,
        "residual_win_rate_pct": round((residual_wins / total_runs) * 100.0, 1) if total_runs > 0 else 0.0,
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
            "mean": round(float(np.mean(direct_maes)), 3),
            "std": round(float(np.std(direct_maes)), 3),
            "min": round(float(np.min(direct_maes)), 3),
            "max": round(float(np.max(direct_maes)), 3),
        },
        "ml_rmse": {
            "mean": round(float(np.mean(direct_rmses)), 3),
            "std": round(float(np.std(direct_rmses)), 3),
        },
        "ml_f1": {
            "mean": round(float(np.mean(direct_f1s)), 3),
            "std": round(float(np.std(direct_f1s)), 3),
        },
        "residual_mae": {
            "mean": round(float(np.mean(residual_maes)), 3),
            "std": round(float(np.std(residual_maes)), 3),
            "min": round(float(np.min(residual_maes)), 3),
            "max": round(float(np.max(residual_maes)), 3),
        },
        "residual_rmse": {
            "mean": round(float(np.mean(residual_rmses)), 3),
            "std": round(float(np.std(residual_rmses)), 3),
        },
        "residual_precision": {
            "mean": round(float(np.mean(residual_precs)), 3),
            "std": round(float(np.std(residual_precs)), 3),
        },
        "residual_recall": {
            "mean": round(float(np.mean(residual_recs)), 3),
            "std": round(float(np.std(residual_recs)), 3),
        },
        "residual_f1": {
            "mean": round(float(np.mean(residual_f1s)), 3),
            "std": round(float(np.std(residual_f1s)), 3),
        },
        "direct_mae_improvement_pct": {
            "mean": round(float(np.mean(direct_mae_impvs)), 3),
            "std": round(float(np.std(direct_mae_impvs)), 3),
        },
        "direct_rmse_improvement_pct": {
            "mean": round(float(np.mean(direct_rmse_impvs)), 3),
            "std": round(float(np.std(direct_rmse_impvs)), 3),
        },
        "residual_mae_improvement_pct": {
            "mean": round(float(np.mean(residual_mae_impvs)), 3),
            "std": round(float(np.std(residual_mae_impvs)), 3),
        },
        "residual_rmse_improvement_pct": {
            "mean": round(float(np.mean(residual_rmse_impvs)), 3),
            "std": round(float(np.std(residual_rmse_impvs)), 3),
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
        "=" * 84,
        f" REPEATED-SEED EVALUATION SUMMARY ({s['num_runs']} Deterministic Runs)",
        "=" * 84,
        f"Seeds Evaluated: {s['seeds_evaluated']}",
        f"Direct ML Won vs Baseline   : {s['model_beat_baseline_mae_count']} / {s['num_runs']} ({s['win_rate_pct']}%)",
        f"Residual ML Won vs Baseline : {s['residual_beat_baseline_mae_count']} / {s['num_runs']} ({s['residual_win_rate_pct']}%)",
        "-" * 84,
        f"{'Metric':<22} | {'Scheduled Baseline':<20} | {'Direct ML Model':<20} | {'Residual ML Model':<20}",
        "-" * 84,
        f"{'MAE (hours)':<22} | {s['baseline_mae']['mean']:>5.3f} +/- {s['baseline_mae']['std']:<9.3f} | {s['ml_mae']['mean']:>5.3f} +/- {s['ml_mae']['std']:<9.3f} | {s['residual_mae']['mean']:>5.3f} +/- {s['residual_mae']['std']:<9.3f}",
        f"{'RMSE (hours)':<22} | {s['baseline_rmse']['mean']:>5.3f} +/- {s['baseline_rmse']['std']:<9.3f} | {s['ml_rmse']['mean']:>5.3f} +/- {s['ml_rmse']['std']:<9.3f} | {s['residual_rmse']['mean']:>5.3f} +/- {s['residual_rmse']['std']:<9.3f}",
        f"{'Delay Risk F1':<22} | {s['baseline_f1']['mean']:>5.3f} +/- {s['baseline_f1']['std']:<9.3f} | {s['ml_f1']['mean']:>5.3f} +/- {s['ml_f1']['std']:<9.3f} | {s['residual_f1']['mean']:>5.3f} +/- {s['residual_f1']['std']:<9.3f}",
        "-" * 84,
        f"Residual Mean MAE Improvement  : +{s['residual_mae_improvement_pct']['mean']:.1f}% (std: {s['residual_mae_improvement_pct']['std']:.1f}%)",
        f"Residual Mean RMSE Improvement : +{s['residual_rmse_improvement_pct']['mean']:.1f}% (std: {s['residual_rmse_improvement_pct']['std']:.1f}%)",
        "-" * 84,
        f"Note: {s['methodology_disclaimer']}",
        "=" * 84,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_repeated_seed_evaluation()
    print(format_repeated_seed_summary(res))
