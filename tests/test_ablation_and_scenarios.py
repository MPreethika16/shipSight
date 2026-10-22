"""Automated tests for customs feature ablation and disruption scenario analysis.
Verifies evaluation consistency, degradation assertions, reproducibility,
and correct ground-truth / input modifications.
"""

import unittest

from src.ablation_and_scenarios import (
    evaluate_customs_ablation,
    run_controlled_disruption_scenario,
)


class TestAblationAndScenarios(unittest.TestCase):
    def test_customs_ablation_metrics_and_degradation(self):
        res = evaluate_customs_ablation()

        self.assertIn("comparison", res)
        c = res["comparison"]

        # Assert full model outperforms ablated model
        self.assertLess(c["mae_full"], c["mae_ablated"])
        self.assertLess(c["rmse_full"], c["rmse_ablated"])
        self.assertGreater(c["f1_full"], c["f1_ablated"])

        # Assert substantial degradation percentage
        self.assertGreater(c["mae_degradation_pct"], 50.0)

        # Baseline check
        self.assertAlmostEqual(c["mae_full"], 5.921, places=2)
        self.assertAlmostEqual(c["rmse_full"], 7.237, places=2)

    def test_disruption_scenario_reproducibility(self):
        run1 = run_controlled_disruption_scenario(congestion_input_delta=0.35, customs_delay_hours=24.0)
        run2 = run_controlled_disruption_scenario(congestion_input_delta=0.35, customs_delay_hours=24.0)

        s1_run1 = run1["input_sensitivity_scenario"]["mean_forecast_increase_hours"]
        s1_run2 = run2["input_sensitivity_scenario"]["mean_forecast_increase_hours"]
        self.assertEqual(s1_run1, s1_run2)

        s2_run1 = run1["ground_truth_disruption_scenario"]["disrupted_model_mae"]
        s2_run2 = run2["ground_truth_disruption_scenario"]["disrupted_model_mae"]
        self.assertEqual(s2_run1, s2_run2)

    def test_disruption_scenario_logic(self):
        res = run_controlled_disruption_scenario(congestion_input_delta=0.35, customs_delay_hours=24.0)

        s1 = res["input_sensitivity_scenario"]
        s2 = res["ground_truth_disruption_scenario"]

        # Increasing congestion should increase remaining duration forecast
        self.assertGreater(s1["mean_forecast_increase_hours"], 0.0)

        # Prolonging customs should increase model test error
        self.assertGreater(s2["disrupted_model_mae"], s2["normal_mae"])
        self.assertGreater(s2["model_error_increase_hours"], 0.0)

        # Required disclaimer check
        self.assertIn("disclaimer", res)
        self.assertIn("synthetic simulation", res["disclaimer"])


if __name__ == "__main__":
    unittest.main()
