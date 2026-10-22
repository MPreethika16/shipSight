"""Master executable pipeline for ShipSight.
Runs end-to-end data generation, automated test verification, baseline vs ML evaluation,
and delay-propagation analytics.
"""

from __future__ import annotations

import sys
import unittest

from src.generator import export_synthetic_csv, generate_synthetic_shipments
from src.model import run_training_and_evaluation
from src.propagation_analysis import analyze_delay_propagation
from src.repeated_seed_evaluation import (
    format_repeated_seed_summary,
    run_repeated_seed_evaluation,
)


def main() -> int:
    print("=" * 70)
    print(" SHIPSIGHT: EXPLAINABLE ETA AND MULTI-STAGE DELAY PROPAGATION")
    print("=" * 70)

    # 1. Generate Synthetic Dataset
    print("\n[Step 1/6] Generating 250-row synthetic shipment dataset...")
    records = generate_synthetic_shipments(num_shipments=250, seed=42)
    csv_path = export_synthetic_csv(records)
    print(f"  -> Generated {len(records)} records saved to {csv_path}")

    # 2. Run Automated Verification Tests
    print("\n[Step 2/6] Running automated pipeline integrity test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)

    if not test_result.wasSuccessful():
        print("  -> ERROR: Pipeline unit tests failed! Aborting execution.")
        return 1
    print(f"  -> All {test_result.testsRun} unit tests passed successfully!")

    # 3. Model Training & Evaluation
    print("\n[Step 3/6] Training and evaluating baseline vs HistGradientBoostingRegressor...")
    model_results = run_training_and_evaluation()

    base = model_results["baseline"]
    ml = model_results["hist_gradient_boosting"]

    print("\n" + "-" * 70)
    print(" EMPIRICAL EVALUATION METRICS (Chronological Held-Out Test Set, N=50)")
    print("-" * 70)
    print(f"{'Metric':<32} | {'Scheduled Baseline':<16} | {'HistGradientBoosting':<16} | {'Improvement':<12}")
    print("-" * 70)
    mae_diff = (base["mae_hours"] - ml["mae_hours"]) / base["mae_hours"] * 100
    rmse_diff = (base["rmse_hours"] - ml["rmse_hours"]) / base["rmse_hours"] * 100
    print(f"{'MAE (hours)':<32} | {base['mae_hours']:<16.3f} | {ml['mae_hours']:<16.3f} | -{mae_diff:.1f}%")
    print(f"{'RMSE (hours)':<32} | {base['rmse_hours']:<16.3f} | {ml['rmse_hours']:<16.3f} | -{rmse_diff:.1f}%")
    print(f"{'MAPE (%) [Caveat: zero-skew]':<32} | {base['mape_pct']:<16.2f} | {ml['mape_pct']:<16.2f} | -")
    print(f"{'Delay Risk F1-Score (tau=6h)':<32} | {base['risk_f1']:<16.3f} | {ml['risk_f1']:<16.3f} | -")
    print(f"{'Delay Risk Precision':<32} | {base['risk_precision']:<16.3f} | {ml['risk_precision']:<16.3f} | -")
    print(f"{'Delay Risk Recall':<32} | {base['risk_recall']:<16.3f} | {ml['risk_recall']:<16.3f} | -")
    print("-" * 70)

    print("\nTop Feature Attributions (Permutation MAE Impact when Shuffled on Test Set):")
    for feat in model_results["feature_importances"][:6]:
        print(f"  * {feat['feature']:<28}: +{feat['mean_mae_increase']:.3f} hrs error (std: {feat['std']:.3f})")

    # 4. Repeated-Seed Stability Evaluation
    print("\n[Step 4/6] Evaluating pipeline stability across 10 deterministic random seeds...")
    seed_results = run_repeated_seed_evaluation()
    print("\n" + format_repeated_seed_summary(seed_results))

    # 5. Delay Propagation Analysis
    print("\n[Step 5/6] Analyzing stage-by-stage delay propagation dynamics...")
    prop_results = analyze_delay_propagation(records)

    print("\nEmpirical Delay Transmission Slopes across Stages (Delta D_k / Delta D_{k-1}):")
    for transition, slope in prop_results["transmission_slopes"].items():
        print(f"  * {transition:<45}: {slope:.3f}")

    print("\nCustoms Disruption Impact:")
    print(f"  * Inspected Containers ({prop_results['customs_impact']['inspected_count']} shipments): Mean dwell = {prop_results['customs_impact']['avg_customs_hours_inspected']} hrs")
    print(f"  * Auto-Cleared Containers ({prop_results['customs_impact']['auto_cleared_count']} shipments): Mean dwell = {prop_results['customs_impact']['avg_customs_hours_auto_cleared']} hrs")
    print(f"  * Net Disruption Penalty: +{prop_results['customs_impact']['added_delay_from_inspection_hrs']} hrs")

    # 6. Live Prediction Explainer & Retrospective Evaluation Demo
    print("\n[Step 6/6] Demonstrating Live Prediction Explainer & Retrospective Audit...")
    from src.model import split_chronological
    from src.shipment_explainer import explain_shipment, format_shipment_explanation

    _, test_records = split_chronological(records)
    sample_shipment = test_records[0]
    pipeline = model_results["pipeline"]

    print("\n" + "-" * 70)
    print(" 6A. LIVE OPERATIONAL PREDICTION (Strict Pre-Cutoff Quarantine)")
    print("-" * 70)
    live_explanation = explain_shipment(sample_shipment, pipeline, mode="live")
    print(format_shipment_explanation(live_explanation))

    print("\n" + "-" * 70)
    print(" 6B. RETROSPECTIVE EVALUATION AUDIT (Post-Delivery Evaluation)")
    print("-" * 70)
    retro_explanation = explain_shipment(sample_shipment, pipeline, mode="retrospective")
    print(format_shipment_explanation(retro_explanation))

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY WITH ZERO FAILURES")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
