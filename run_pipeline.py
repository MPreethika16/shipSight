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
    print("\n[Step 3/6] Training and evaluating baseline vs Direct vs Residual GBDT...")
    model_results = run_training_and_evaluation()

    base = model_results["baseline"]
    ml = model_results["hist_gradient_boosting"]
    res_m = model_results["residual_model"]

    print("\n" + "-" * 88)
    print(" EMPIRICAL EVALUATION METRICS (Chronological Held-Out Test Set, N=50)")
    print("-" * 88)
    print(f"{'Metric':<28} | {'Scheduled Base':<15} | {'Direct GBDT':<15} | {'Residual GBDT':<15} | {'Res vs Base':<12}")
    print("-" * 88)
    res_mae_diff = (base["mae_hours"] - res_m["mae_hours"]) / base["mae_hours"] * 100
    res_rmse_diff = (base["rmse_hours"] - res_m["rmse_hours"]) / base["rmse_hours"] * 100
    print(f"{'MAE (hours)':<28} | {base['mae_hours']:<15.3f} | {ml['mae_hours']:<15.3f} | {res_m['mae_hours']:<15.3f} | -{res_mae_diff:.1f}%")
    print(f"{'RMSE (hours)':<28} | {base['rmse_hours']:<15.3f} | {ml['rmse_hours']:<15.3f} | {res_m['rmse_hours']:<15.3f} | -{res_rmse_diff:.1f}%")
    print(f"{'MAPE (%) [zero-skew]':<28} | {base['mape_pct']:<15.2f} | {ml['mape_pct']:<15.2f} | {res_m['mape_pct']:<15.2f} | -")
    print(f"{'Delay Risk F1-Score (6h)':<28} | {base['risk_f1']:<15.3f} | {ml['risk_f1']:<15.3f} | {res_m['risk_f1']:<15.3f} | -")
    print(f"{'Delay Risk Precision':<28} | {base['risk_precision']:<15.3f} | {ml['risk_precision']:<15.3f} | {res_m['risk_precision']:<15.3f} | -")
    print(f"{'Delay Risk Recall':<28} | {base['risk_recall']:<15.3f} | {ml['risk_recall']:<15.3f} | {res_m['risk_recall']:<15.3f} | -")
    print("-" * 88)

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
    predictor = model_results["residual_predictor"]

    print("\n" + "-" * 70)
    print(" 6A. LIVE OPERATIONAL PREDICTION (Strict Pre-Cutoff Quarantine - Residual GBDT)")
    print("-" * 70)
    live_explanation = explain_shipment(sample_shipment, predictor, mode="live")
    print(format_shipment_explanation(live_explanation))

    print("\n" + "-" * 70)
    print(" 6B. RETROSPECTIVE EVALUATION AUDIT (Post-Delivery Evaluation - Residual GBDT)")
    print("-" * 70)
    retro_explanation = explain_shipment(sample_shipment, predictor, mode="retrospective")
    print(format_shipment_explanation(retro_explanation))

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY WITH ZERO FAILURES")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
