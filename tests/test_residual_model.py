"""Automated test suite for Residual ETA prediction model.

Verifies:
1. Target arithmetic: residual_target = actual_remaining - scheduled_remaining.
2. Prediction reconstruction: predicted_remaining = max(0.0, scheduled_remaining + predicted_residual).
3. Non-negative clipping under extreme negative residual perturbations.
4. Zero feature leakage and downstream immutability.
5. Backward compatibility with existing pipeline results, metrics, and interfaces.
6. Explanation module integration, invariant delay arithmetic, and live-mode data quarantine.
"""

from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    DELAY_RISK_THRESHOLD_HOURS,
    FEATURE_WHITELIST,
    NUM_SHIPMENTS,
    PROHIBITED_LEAKAGE_COLUMNS,
)
from src.generator import generate_synthetic_shipments
from src.model import (
    ResidualETAPredictor,
    build_feature_matrix,
    calculate_scheduled_remaining_duration,
    run_training_and_evaluation,
    split_chronological,
)
from src.shipment_explainer import (
    calculate_scenario_sensitivity,
    explain_shipment,
    format_shipment_explanation,
)


class TestResidualETAModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_synthetic_shipments(num_shipments=NUM_SHIPMENTS, seed=42)
        cls.train_rec, cls.test_rec = split_chronological(cls.records)
        cls.X_train_dicts, cls.y_train_rem, _, _ = build_feature_matrix(cls.train_rec)
        cls.X_test_dicts, cls.y_test_rem, _, _ = build_feature_matrix(cls.test_rec)

        cls.y_train_sched = calculate_scheduled_remaining_duration(cls.train_rec)
        cls.y_test_sched = calculate_scheduled_remaining_duration(cls.test_rec)

        cls.df_X_train = pd.DataFrame(cls.X_train_dicts)
        cls.df_X_test = pd.DataFrame(cls.X_test_dicts)

        cls.predictor = ResidualETAPredictor()
        cls.predictor.fit(cls.df_X_train, cls.y_train_rem, cls.y_train_sched)

    def test_residual_target_arithmetic(self):
        """Verify residual_target == actual_remaining_hours - scheduled_remaining_hours exactly."""
        y_residual_direct = self.y_train_rem - self.y_train_sched
        for i, r in enumerate(self.train_rec):
            sched_sum = (
                r["scheduled_terminal_hours"]
                + r["scheduled_customs_hours"]
                + r["scheduled_drayage_hours"]
            )
            expected_residual = r["remaining_duration_hours"] - sched_sum
            self.assertAlmostEqual(y_residual_direct[i], expected_residual, places=6)

    def test_prediction_reconstruction(self):
        """Verify predicted_remaining == max(0.0, scheduled_remaining + predicted_residual)."""
        pred_res = self.predictor.predict_residual(self.df_X_test)
        pred_rem_with_sched = self.predictor.predict(self.df_X_test, self.y_test_sched)
        pred_rem_auto_sched = self.predictor.predict(self.df_X_test)

        # Both explicit and automatic sched parameter must produce identical results
        np.testing.assert_allclose(pred_rem_with_sched, pred_rem_auto_sched, rtol=1e-6)

        expected_rem = np.maximum(0.0, self.y_test_sched + pred_res)
        np.testing.assert_allclose(pred_rem_with_sched, expected_rem, rtol=1e-6)

    def test_nonnegative_clipping_under_extreme_negative_residual(self):
        """Assert that extreme negative predicted residuals are clipped to >= 0.0."""
        # Mock negative residual by manually feeding small scheduled hours
        tiny_sched = np.array([5.0] * len(self.df_X_test))
        # If predicted residual is -20.0, sum is -15.0 -> must clip to 0.0
        synthetic_preds = np.maximum(0.0, tiny_sched + (-20.0))
        self.assertTrue(np.all(synthetic_preds >= 0.0))
        self.assertEqual(synthetic_preds[0], 0.0)

        # Test predict method with an artificially small scheduled remaining array
        clipped_preds = self.predictor.predict(self.df_X_test, tiny_sched - 100.0)
        self.assertTrue(np.all(clipped_preds >= 0.0))
        self.assertTrue(np.all(clipped_preds == 0.0))

    def test_feature_leakage_and_admissibility(self):
        """Assert zero prohibited downstream columns in feature matrix and invariance to post-cutoff tampering."""
        # 1. Assert no leakage columns in feature matrix
        for prohibited in PROHIBITED_LEAKAGE_COLUMNS:
            self.assertNotIn(prohibited, self.df_X_train.columns)
            self.assertNotIn(prohibited, self.df_X_test.columns)

        # 2. Mutate downstream actuals in raw records and verify predictions remain perfectly invariant
        tampered_rec = copy.deepcopy(self.test_rec)
        for r in tampered_rec:
            r["actual_terminal_hours"] = 999.0
            r["actual_customs_hours"] = 888.0
            r["actual_drayage_hours"] = 777.0
            r["remaining_duration_hours"] = 2664.0
            r["final_delay_hours"] = 2500.0
            r["actual_final_delivery_time"] = "2026-12-31T23:59:59"

        X_tampered_dicts, _, _, _ = build_feature_matrix(tampered_rec)
        df_tampered = pd.DataFrame(X_tampered_dicts)

        preds_orig = self.predictor.predict(self.df_X_test)
        preds_tampered = self.predictor.predict(df_tampered)
        np.testing.assert_allclose(preds_orig, preds_tampered, rtol=1e-9)

    def test_backward_compatibility(self):
        """Assert run_training_and_evaluation() returns original keys alongside residual results."""
        results = run_training_and_evaluation()

        # Direct model keys must remain intact
        self.assertIn("baseline", results)
        self.assertIn("model", results)
        metric_keys = ["mae_hours", "rmse_hours", "mape_pct", "risk_precision", "risk_recall", "risk_f1", "confusion_matrix"]
        for k in metric_keys:
            self.assertIn(k, results["baseline"])
            self.assertIn(k, results["model"])

        # Residual model keys must be present
        self.assertIn("residual_model", results)
        for k in metric_keys:
            self.assertIn(k, results["residual_model"])

        # Estimator objects available
        self.assertIn("pipeline", results)
        self.assertIn("residual_predictor", results)
        self.assertTrue(isinstance(results["residual_predictor"], ResidualETAPredictor))

    def test_explainer_integration_and_invariants(self):
        """Verify explanation module reconstructs ETA and delay with ResidualETAPredictor."""
        sample = self.records[12]
        exp = explain_shipment(sample, self.predictor, mode="live")

        d = exp["durations_and_delays"]
        t = exp["timestamps"]
        r = exp["delay_risk"]

        # Invariant 1: ETA timestamp matches berth time + remaining duration
        berth_time = datetime.fromisoformat(exp["prediction_cutoff"]["timestamp"])
        eta_time = datetime.fromisoformat(t["predicted_final_eta"])
        expected_eta = berth_time + timedelta(hours=d["predicted_remaining_duration_hours"])
        self.assertAlmostEqual(eta_time.timestamp(), expected_eta.timestamp(), delta=1.0)

        # Invariant 2: Downstream slippage matches predicted remaining - scheduled remaining
        expected_delta = d["predicted_remaining_duration_hours"] - d["scheduled_downstream_allowance_hours"]
        self.assertAlmostEqual(d["predicted_downstream_delta_hours"], expected_delta, places=2)

        # Invariant 3: Residual equals predicted downstream delta
        self.assertAlmostEqual(d["predicted_residual_hours"], d["predicted_downstream_delta_hours"], places=2)

        # Invariant 4: Predicted final delay = carried delay at berth + predicted residual
        expected_final_delay = d["carried_delay_at_berth_hours"] + d["predicted_residual_hours"]
        self.assertAlmostEqual(d["predicted_final_delay_hours"], expected_final_delay, places=2)

        # Invariant 5: Delay vs schedule matches ETA vs scheduled delivery timestamp difference
        sched_delivery = datetime.fromisoformat(t["scheduled_final_delivery"])
        eta_delay_hours = (eta_time - sched_delivery).total_seconds() / 3600.0
        self.assertAlmostEqual(d["predicted_final_delay_hours"], eta_delay_hours, delta=0.02)

        # Risk classification matches threshold
        if d["predicted_final_delay_hours"] > DELAY_RISK_THRESHOLD_HOURS:
            self.assertEqual(r["predicted_risk_binary"], 1)
            self.assertEqual(r["predicted_risk"], "DELAYED")
        else:
            self.assertEqual(r["predicted_risk_binary"], 0)
            self.assertEqual(r["predicted_risk"], "ON-TIME")

    def test_explainer_live_mode_field_allowlist_and_quarantine(self):
        """Verify that injected downstream or unknown keys never leak into live explanation."""
        tampered_sample = copy.deepcopy(self.records[20])
        tampered_sample["secret_downstream_leak"] = "UNAUTHORIZED_LEAK"
        tampered_sample["inspection_note"] = "Customs agent notes physical inspection required"
        tampered_sample["actual_terminal_hours"] = 72.0

        exp_live = explain_shipment(tampered_sample, self.predictor, mode="live")

        # Quarantined keys must not be present anywhere in the live explanation dictionary
        self.assertNotIn("secret_downstream_leak", exp_live)
        self.assertNotIn("inspection_note", exp_live)
        self.assertNotIn("actual_terminal_hours", exp_live)
        self.assertNotIn("actual_final_delivery", exp_live["timestamps"])
        self.assertNotIn("actual_final_delay_hours", exp_live["durations_and_delays"])
        self.assertNotIn("delay_error_hours", exp_live["durations_and_delays"])
        self.assertNotIn("retrospective_evaluation", exp_live)

    def test_explainer_retrospective_mode_evaluation(self):
        """Verify retrospective evaluation correctly evaluates residual predictor against realized ground truth."""
        sample = self.records[25]
        exp_retro = explain_shipment(sample, self.predictor, mode="retrospective")

        self.assertIn("retrospective_evaluation", exp_retro)
        retro = exp_retro["retrospective_evaluation"]
        self.assertIn("actual_final_delay_hours", retro)
        self.assertIn("prediction_error_remaining_hours", retro)
        self.assertIn("prediction_error_final_delay_hours", retro)
        self.assertIn("risk_classification_correct", retro)

    def test_explainer_scenario_sensitivity_with_residual_model(self):
        """Verify scenario sensitivity evaluation with ResidualETAPredictor."""
        sample = self.records[30]
        exp = explain_shipment(sample, self.predictor, mode="live", evaluate_scenarios=True)

        self.assertIn("scenario_sensitivity", exp)
        scen_info = exp["scenario_sensitivity"]
        self.assertIn("scenarios", scen_info)
        self.assertGreater(len(scen_info["scenarios"]), 0)

        for s in scen_info["scenarios"]:
            self.assertIn("scenario_predicted_remaining_hours", s)
            self.assertIn("sensitivity_remaining_delta_hours", s)
            self.assertIn("sensitivity_final_delay_delta_hours", s)
            # Label must be explicit model sensitivity
            self.assertIn("Model Sensitivity", scen_info["label"])
            self.assertIn("Not Causal Attribution", scen_info["label"])


if __name__ == "__main__":
    unittest.main()
