"""Automated tests for shipment-level ETA and delay explanation module.
Verifies calculation invariants, delay deconstruction, risk classification,
and strict absence of downstream target leakage.
"""

import unittest
from datetime import datetime, timedelta

import pandas as pd

from src.config import DELAY_RISK_THRESHOLD_HOURS, NUM_SHIPMENTS
from src.generator import generate_synthetic_shipments
from src.model import build_feature_matrix, build_pipeline, split_chronological
from src.shipment_explainer import explain_shipment, format_shipment_explanation


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
        exp = explain_shipment(sample, self.pipeline)

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

        # 4. Risk classification matches threshold
        if d["predicted_final_delay_hours"] > DELAY_RISK_THRESHOLD_HOURS:
            self.assertEqual(r["predicted_risk_binary"], 1)
            self.assertEqual(r["predicted_risk"], "DELAYED")
        else:
            self.assertEqual(r["predicted_risk_binary"], 0)
            self.assertEqual(r["predicted_risk"], "ON-TIME")

    def test_zero_leakage_from_downstream_actuals(self):
        sample = dict(self.records[20])
        exp_original = explain_shipment(sample, self.pipeline)

        # Mutate realized downstream columns that should NOT be visible
        sample["actual_terminal_hours"] += 50.0
        sample["actual_customs_hours"] += 50.0
        sample["actual_drayage_hours"] += 50.0
        sample["final_delay_hours"] += 150.0

        exp_mutated = explain_shipment(sample, self.pipeline)

        # Predicted remaining duration and delay must remain identical
        self.assertEqual(
            exp_original["durations_and_delays"]["predicted_remaining_duration_hours"],
            exp_mutated["durations_and_delays"]["predicted_remaining_duration_hours"],
        )
        self.assertEqual(
            exp_original["durations_and_delays"]["predicted_final_delay_hours"],
            exp_mutated["durations_and_delays"]["predicted_final_delay_hours"],
        )

    def test_format_explanation_string(self):
        sample = self.records[5]
        exp = explain_shipment(sample, self.pipeline)
        formatted = format_shipment_explanation(exp)

        self.assertIn("SHIPSIGHT INDIVIDUAL SHIPMENT ETA EXPLANATION", formatted)
        self.assertIn(sample["shipment_id"], formatted)
        self.assertIn("DISCLAIMER:", formatted)
        self.assertIn("DELAY DECONSTRUCTION", formatted)


if __name__ == "__main__":
    unittest.main()
