"""Automated tests for shipment-level ETA and delay explanation module.

Verifies:
1. Strict separation of live prediction vs retrospective evaluation modes.
2. Complete absence of downstream actual outcome leakage in live prediction output.
3. Invariant mathematical consistency of predicted ETA and delay arithmetic.
4. Correctness of scenario sensitivity re-prediction via the actual fitted pipeline.
5. Explicit rejection of causal attribution or SHAP claims.
"""

from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta

import pandas as pd

from src.config import DELAY_RISK_THRESHOLD_HOURS, FEATURE_WHITELIST, NUM_SHIPMENTS
from src.generator import generate_synthetic_shipments
from src.model import build_feature_matrix, build_pipeline, split_chronological
from src.shipment_explainer import (
    calculate_scenario_sensitivity,
    explain_shipment,
    format_shipment_explanation,
)


class TestShipmentExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_synthetic_shipments(num_shipments=NUM_SHIPMENTS, seed=42)
        train_rec, _ = split_chronological(cls.records)
        X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_rec)
        cls.pipeline = build_pipeline()
        cls.pipeline.fit(pd.DataFrame(X_train_dicts), y_train_rem)

    def test_calculation_invariants(self):
        sample = self.records[10]
        exp = explain_shipment(sample, self.pipeline, mode="live")

        d = exp["durations_and_delays"]
        t = exp["timestamps"]
        r = exp["delay_risk"]

        # 1. ETA timestamp matches berth time + remaining duration
        berth_time = datetime.fromisoformat(exp["prediction_cutoff"]["timestamp"])
        eta_time = datetime.fromisoformat(t["predicted_final_eta"])
        expected_eta = berth_time + timedelta(hours=d["predicted_remaining_duration_hours"])
        self.assertAlmostEqual(eta_time.timestamp(), expected_eta.timestamp(), delta=1.0)

        # 2. Downstream delta = predicted remaining - scheduled downstream
        expected_delta = d["predicted_remaining_duration_hours"] - d["scheduled_downstream_allowance_hours"]
        self.assertAlmostEqual(d["predicted_downstream_delta_hours"], expected_delta, places=2)

        # 3. Final delay = carried delay at berth + downstream delta
        expected_final_delay = d["carried_delay_at_berth_hours"] + d["predicted_downstream_delta_hours"]
        self.assertAlmostEqual(d["predicted_final_delay_hours"], expected_final_delay, places=2)
        self.assertEqual(d["predicted_final_delay_hours"], d["predicted_delay_relative_to_schedule_hours"])

        # 4. Invariant: predicted delay vs schedule equals (predicted_final_eta - scheduled_final_delivery) in hours
        sched_delivery_time = datetime.fromisoformat(t["scheduled_final_delivery"])
        eta_delay_hours = (eta_time - sched_delivery_time).total_seconds() / 3600.0
        self.assertAlmostEqual(d["predicted_final_delay_hours"], eta_delay_hours, delta=0.02)

        # 5. Risk classification matches threshold
        if d["predicted_final_delay_hours"] > DELAY_RISK_THRESHOLD_HOURS:
            self.assertEqual(r["predicted_risk_binary"], 1)
            self.assertEqual(r["predicted_risk"], "DELAYED")
        else:
            self.assertEqual(r["predicted_risk_binary"], 0)
            self.assertEqual(r["predicted_risk"], "ON-TIME")

    def test_live_mode_strict_absence_of_downstream_actuals(self):
        """Confirm that live mode output never reveals actual final delivery or realized delays."""
        sample = self.records[15]
        exp_live = explain_shipment(sample, self.pipeline, mode="live")

        self.assertEqual(exp_live["mode"], "live")

        # In live mode, downstream actuals must NOT exist in the explanation dictionary
        self.assertNotIn("actual_final_delivery", exp_live["timestamps"])
        self.assertNotIn("actual_final_delay_hours", exp_live["durations_and_delays"])
        self.assertNotIn("delay_error_hours", exp_live["durations_and_delays"])
        self.assertNotIn("actual_risk_binary", exp_live["delay_risk"])
        self.assertNotIn("retrospective_evaluation", exp_live)

        # Formatted live report must never reveal actual delivery or forecast error
        formatted_live = format_shipment_explanation(exp_live)
        self.assertIn("[LIVE OPERATIONAL FORECAST]", formatted_live)
        self.assertNotIn("Actual Final Delivery", formatted_live)
        self.assertNotIn("Realized Actual Final Delay", formatted_live)
        self.assertNotIn("Forecast Error", formatted_live)
        self.assertNotIn("Actual Realized Risk", formatted_live)

    def test_zero_leakage_from_downstream_actuals_under_mutation(self):
        """Downstream actual outcomes cannot influence live predictions."""
        sample = copy.deepcopy(self.records[20])
        exp_original = explain_shipment(sample, self.pipeline, mode="live")

        # Mutate downstream realization columns that occur AFTER vessel berthing
        sample["actual_terminal_hours"] += 72.0
        sample["actual_customs_hours"] += 48.0
        sample["actual_drayage_hours"] += 12.0
        sample["final_delay_hours"] += 132.0
        sample["remaining_duration_hours"] += 132.0
        sample["delay_risk_binary"] = 1
        sample["actual_final_delivery_time"] = "2026-12-31T23:59:59"

        exp_mutated = explain_shipment(sample, self.pipeline, mode="live")

        # Predictions, delays, and scenario sensitivities must be completely identical
        self.assertEqual(
            exp_original["durations_and_delays"]["predicted_remaining_duration_hours"],
            exp_mutated["durations_and_delays"]["predicted_remaining_duration_hours"],
        )
        self.assertEqual(
            exp_original["durations_and_delays"]["predicted_final_delay_hours"],
            exp_mutated["durations_and_delays"]["predicted_final_delay_hours"],
        )
        self.assertEqual(
            exp_original["timestamps"]["predicted_final_eta"],
            exp_mutated["timestamps"]["predicted_final_eta"],
        )
        self.assertEqual(
            exp_original["scenario_sensitivity"]["scenarios"],
            exp_mutated["scenario_sensitivity"]["scenarios"],
        )

    def test_injected_unknown_keys_and_notes_cannot_appear_in_live_output(self):
        """Confirm that unknown downstream keys, notes, or injected telemetry cannot leak into live output."""
        sample = copy.deepcopy(self.records[7])

        # Inject various unknown/downstream keys, unvetted notes, and unexpected fields
        injected_fields = {
            "inspection_note": "Advance administrative hold flag known at berth; physical inspection expected",
            "unvetted_customs_note": "Confidential physical inspection note",
            "actual_terminal_dwell_hours": 99.9,
            "downstream_gate_out_timestamp": "2026-05-01T10:00:00",
            "customs_hold_officer_id": "OFFICER_42",
            "actual_drayage_traffic_jam_hours": 15.0,
            "random_future_telemetry_metric": 42.42,
        }
        for k, v in injected_fields.items():
            sample[k] = v

        exp_live = explain_shipment(sample, self.pipeline, mode="live")
        formatted_live = format_shipment_explanation(exp_live)

        # 1. Assert none of the injected keys or values appear anywhere in the live explanation dictionary
        exp_live_str = str(exp_live)
        for k, v in injected_fields.items():
            self.assertNotIn(k, exp_live, f"Injected key '{k}' appeared at top-level of live explanation dict")
            self.assertNotIn(str(v), exp_live_str, f"Injected value for '{k}' appeared inside live explanation dict")

        # 2. Assert none of the injected keys or values appear in the formatted report
        for k, v in injected_fields.items():
            self.assertNotIn(k, formatted_live, f"Injected key '{k}' leaked into live formatted output")
            self.assertNotIn(str(v), formatted_live, f"Injected value for '{k}' leaked into live formatted output")

    def test_retrospective_mode_evaluation_integrity(self):
        """Confirm retrospective evaluation reports realized outcomes, forecast errors, and risk audit."""
        sample = self.records[8]
        exp_retro = explain_shipment(sample, self.pipeline, mode="retrospective")

        self.assertEqual(exp_retro["mode"], "retrospective")
        self.assertIn("actual_final_delivery", exp_retro["timestamps"])
        self.assertIn("actual_final_delay_hours", exp_retro["durations_and_delays"])
        self.assertIn("delay_error_hours", exp_retro["durations_and_delays"])
        self.assertIn("retrospective_evaluation", exp_retro)

        retro = exp_retro["retrospective_evaluation"]
        self.assertEqual(retro["actual_final_delivery_time"], sample["actual_final_delivery_time"])
        self.assertAlmostEqual(retro["actual_final_delay_hours"], sample["final_delay_hours"], places=2)

        expected_error = round(
            exp_retro["durations_and_delays"]["predicted_final_delay_hours"] - sample["final_delay_hours"], 2
        )
        self.assertAlmostEqual(retro["prediction_error_final_delay_hours"], expected_error, places=2)

        formatted_retro = format_shipment_explanation(exp_retro)
        self.assertIn("[RETROSPECTIVE EVALUATION]", formatted_retro)
        self.assertIn("Actual Final Delivery", formatted_retro)
        self.assertIn("Realized Actual Final Delay", formatted_retro)
        self.assertIn("Forecast Error", formatted_retro)

    def test_retrospective_mode_requires_ground_truth(self):
        """Confirm that retrospective mode raises ValueError if actual delivery is missing."""
        sample_missing = copy.deepcopy(self.records[3])
        del sample_missing["actual_final_delivery_time"]
        with self.assertRaises(ValueError):
            explain_shipment(sample_missing, self.pipeline, mode="retrospective")

    def test_invalid_mode_raises_value_error(self):
        """Confirm that unsupported mode strings raise ValueError."""
        sample = self.records[0]
        with self.assertRaises(ValueError):
            explain_shipment(sample, self.pipeline, mode="invalid_mode")

    def test_scenario_sensitivity_generated_by_fitted_model(self):
        """Verify scenario predictions are produced by the actual fitted pipeline, not heuristics."""
        sample = self.records[12]
        exp = explain_shipment(sample, self.pipeline, mode="live", evaluate_scenarios=True)

        self.assertIn("scenario_sensitivity", exp)
        sens = exp["scenario_sensitivity"]

        # Confirm non-causal disclaimer and no mention of SHAP
        self.assertIn("Not Causal Attribution", sens["label"])
        self.assertIn("NOT physical causal attributions", sens["interpretation_warning"])
        self.assertIn("MUST NOT be interpreted as additive SHAP", sens["interpretation_warning"])

        scenarios = sens["scenarios"]
        self.assertGreaterEqual(len(scenarios), 1)

        # Verify Scenario 1: customs_inspection_flag toggle
        customs_scens = [s for s in scenarios if s["scenario_name"] == "toggle_customs_inspection_flag"]
        self.assertEqual(len(customs_scens), 1)
        s1 = customs_scens[0]

        # Re-run pipeline directly on modified feature vector and assert exact match
        feat_dict = {col: sample[col] for col in FEATURE_WHITELIST}
        orig_flag = feat_dict["customs_inspection_flag"]
        toggled_flag = 1 - orig_flag
        feat_dict["customs_inspection_flag"] = toggled_flag

        direct_pipeline_pred = float(self.pipeline.predict(pd.DataFrame([feat_dict]))[0])
        self.assertAlmostEqual(s1["scenario_predicted_remaining_hours"], direct_pipeline_pred, places=2)

        # Verify sensitivity delta arithmetic
        expected_rem_delta = round(
            s1["scenario_predicted_remaining_hours"]
            - exp["durations_and_delays"]["predicted_remaining_duration_hours"],
            2,
        )
        self.assertAlmostEqual(s1["sensitivity_remaining_delta_hours"], expected_rem_delta, places=2)
        expected_delay_delta = round(
            s1["scenario_predicted_final_delay_hours"]
            - exp["durations_and_delays"]["predicted_final_delay_hours"],
            2,
        )
        self.assertAlmostEqual(s1["sensitivity_final_delay_delta_hours"], expected_delay_delta, places=2)

    def test_format_explanation_string(self):
        sample = self.records[5]
        exp = explain_shipment(sample, self.pipeline, mode="live")
        formatted = format_shipment_explanation(exp)

        self.assertIn("SHIPSIGHT INDIVIDUAL SHIPMENT ETA EXPLANATION", formatted)
        self.assertIn(sample["shipment_id"], formatted)
        self.assertIn("DISCLAIMER:", formatted)
        self.assertIn("DELAY DECONSTRUCTION", formatted)
        self.assertIn("SCENARIO SENSITIVITY", formatted)
        self.assertIn("toggle_customs_inspection_flag", formatted)


if __name__ == "__main__":
    unittest.main()
