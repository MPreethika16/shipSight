"""Automated tests for repeated-seed evaluation.
Verifies determinism across random seeds, strict chronological split preservation,
baseline calculation consistency, aggregate statistical metrics, and win-count logic.
"""

from datetime import datetime
import unittest
import numpy as np

from src.generator import generate_synthetic_shipments
from src.model import split_chronological
from src.repeated_seed_evaluation import (
    evaluate_single_seed,
    format_repeated_seed_summary,
    run_repeated_seed_evaluation,
)


class TestRepeatedSeeds(unittest.TestCase):
    def test_seed_determinism(self):
        """Verifies that the same seed produces identical dataset and evaluation metrics."""
        res1 = evaluate_single_seed(seed=42, num_shipments=100)
        res2 = evaluate_single_seed(seed=42, num_shipments=100)

        self.assertEqual(res1["seed"], res2["seed"])
        self.assertEqual(res1["baseline_mae"], res2["baseline_mae"])
        self.assertEqual(res1["ml_mae"], res2["ml_mae"])
        self.assertEqual(res1["baseline_rmse"], res2["baseline_rmse"])
        self.assertEqual(res1["ml_rmse"], res2["ml_rmse"])
        self.assertEqual(res1["ml_f1"], res2["ml_f1"])

    def test_chronological_split_integrity_across_seeds(self):
        """Verifies that every seed strictly preserves chronological order without temporal leakage."""
        test_seeds = [42, 123, 999]
        for s in test_seeds:
            records = generate_synthetic_shipments(num_shipments=100, seed=s)
            train_rec, test_rec = split_chronological(records, train_ratio=0.80)

            self.assertEqual(len(train_rec), 80)
            self.assertEqual(len(test_rec), 20)

            # Ensure chronological order
            train_max_dep = max(datetime.fromisoformat(r["scheduled_departure_time"]) for r in train_rec)
            test_min_dep = min(datetime.fromisoformat(r["scheduled_departure_time"]) for r in test_rec)
            self.assertLessEqual(
                train_max_dep,
                test_min_dep,
                f"Temporal leakage detected in seed {s}: train_max ({train_max_dep}) > test_min ({test_min_dep})",
            )

    def test_baseline_formula_consistency(self):
        """Verifies scheduled-duration baseline strictly sums downstream scheduled allowances."""
        records = generate_synthetic_shipments(num_shipments=50, seed=777)
        _, test_rec = split_chronological(records, train_ratio=0.80)

        for r in test_rec:
            expected_sched_downstream = (
                r["scheduled_terminal_hours"]
                + r["scheduled_customs_hours"]
                + r["scheduled_drayage_hours"]
            )
            self.assertGreater(expected_sched_downstream, 0.0)

    def test_aggregate_calculations_and_win_rate(self):
        """Verifies summary statistics (mean, std, win count) across multiple seeds."""
        test_seeds = [42, 101, 202]
        res = run_repeated_seed_evaluation(seeds=test_seeds, num_shipments=100)

        self.assertIn("runs", res)
        self.assertIn("summary", res)
        self.assertEqual(len(res["runs"]), len(test_seeds))

        s = res["summary"]
        self.assertEqual(s["num_runs"], 3)
        self.assertEqual(s["seeds_evaluated"], test_seeds)

        # Manually compute statistics from individual runs
        manual_ml_maes = [r["ml_mae"] for r in res["runs"]]
        expected_mean = round(float(np.mean(manual_ml_maes)), 3)
        expected_std = round(float(np.std(manual_ml_maes)), 3)

        self.assertAlmostEqual(s["ml_mae"]["mean"], expected_mean, places=3)
        self.assertAlmostEqual(s["ml_mae"]["std"], expected_std, places=3)
        self.assertEqual(s["ml_mae"]["min"], min(manual_ml_maes))
        self.assertEqual(s["ml_mae"]["max"], max(manual_ml_maes))

        # Check win count logic
        manual_wins = sum(1 for r in res["runs"] if r["ml_mae"] < r["baseline_mae"])
        self.assertEqual(s["model_beat_baseline_mae_count"], manual_wins)
        self.assertAlmostEqual(s["win_rate_pct"], round(manual_wins / 3 * 100.0, 1))

        # Verify methodology disclaimer
        self.assertIn("methodology_disclaimer", s)
        self.assertIn("synthetic simulations", s["methodology_disclaimer"])

        # Check formatted output string
        formatted = format_repeated_seed_summary(res)
        self.assertIn("REPEATED-SEED EVALUATION SUMMARY", formatted)
        self.assertIn("MAE (hours)", formatted)


if __name__ == "__main__":
    unittest.main()
