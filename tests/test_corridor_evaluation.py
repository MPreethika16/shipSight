"""Automated tests for Leave-One-Corridor-Out (LOCO) evaluation.
Verifies corridor separation, split integrity, zero feature leakage,
and metric consistency across held-out corridors.
"""

import unittest

from src.config import CORRIDORS, NUM_SHIPMENTS
from src.corridor_evaluation import (
    run_leave_one_corridor_out_evaluation,
    split_leave_one_corridor_out,
)
from src.generator import generate_synthetic_shipments


class TestCorridorEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_synthetic_shipments(num_shipments=NUM_SHIPMENTS, seed=42)

    def test_strict_corridor_isolation(self):
        for corridor in CORRIDORS.keys():
            train_rec, test_rec = split_leave_one_corridor_out(self.records, corridor)
            self.assertEqual(len(train_rec) + len(test_rec), len(self.records))
            self.assertTrue(len(test_rec) > 0)
            self.assertTrue(len(train_rec) > 0)

            train_routes = {r["route_id"] for r in train_rec}
            test_routes = {r["route_id"] for r in test_rec}

            self.assertNotIn(corridor, train_routes)
            self.assertEqual(test_routes, {corridor})
            self.assertTrue(train_routes.isdisjoint(test_routes))

    def test_invalid_corridor_raises_error(self):
        with self.assertRaises(ValueError):
            split_leave_one_corridor_out(self.records, "INVALID-ROUTE")

    def test_run_loco_evaluation_output_structure(self):
        results = run_leave_one_corridor_out_evaluation()
        self.assertIn("by_corridor", results)
        self.assertIn("macro_summary", results)

        self.assertEqual(set(results["by_corridor"].keys()), set(CORRIDORS.keys()))
        for corridor, metrics in results["by_corridor"].items():
            self.assertIn("baseline_mae", metrics)
            self.assertIn("baseline_rmse", metrics)
            self.assertIn("model_mae", metrics)
            self.assertIn("model_rmse", metrics)
            self.assertGreater(metrics["baseline_mae"], 0.0)
            self.assertGreater(metrics["model_mae"], 0.0)

        ms = results["macro_summary"]
        self.assertIn("mean_baseline_mae", ms)
        self.assertIn("mean_model_mae", ms)
        self.assertEqual(ms["total_evaluated_records"], len(self.records))


if __name__ == "__main__":
    unittest.main()
