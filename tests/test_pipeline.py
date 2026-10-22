"""Automated test suite for ShipSight pipeline integrity.
Verifies timestamp continuity, nonnegative durations, mathematical target closure,
deterministic generation, threshold-based congestion behavior, feature leakage,
and chronological train/test split.
"""

import unittest
from datetime import datetime
from src.config import (
    CORRIDORS,
    DELAY_RISK_THRESHOLD_HOURS,
    FEATURE_WHITELIST,
    NUM_SHIPMENTS,
    PROHIBITED_LEAKAGE_COLUMNS,
    TRAIN_RATIO,
)
from src.generator import generate_synthetic_shipments, simulate_single_berth_duration


class TestShipSightPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_synthetic_shipments(num_shipments=NUM_SHIPMENTS, seed=42)

    def test_record_count(self):
        self.assertEqual(len(self.records), NUM_SHIPMENTS)

    def test_deterministic_reproducibility(self):
        records_2 = generate_synthetic_shipments(num_shipments=NUM_SHIPMENTS, seed=42)
        self.assertEqual(len(self.records), len(records_2))
        for r1, r2 in zip(self.records, records_2):
            self.assertEqual(r1["shipment_id"], r2["shipment_id"])
            self.assertEqual(r1["actual_final_delivery_time"], r2["actual_final_delivery_time"])
            self.assertAlmostEqual(r1["remaining_duration_hours"], r2["remaining_duration_hours"], places=2)

    def test_port_delay_threshold_logic(self):
        # Case A: Carried delay <= 2.0 hours (reserved slot preserved, zero congestion penalty)
        d_low_cong = simulate_single_berth_duration(
            sched_berth_hours=16.0,
            carried_delay_at_anchorage=1.5,
            port_congestion_index=0.1,
            noise_hours=0.0,
        )
        d_high_cong = simulate_single_berth_duration(
            sched_berth_hours=16.0,
            carried_delay_at_anchorage=1.5,
            port_congestion_index=0.9,
            noise_hours=0.0,
        )
        self.assertEqual(d_low_cong, 16.0)
        self.assertEqual(d_high_cong, 16.0)
        self.assertEqual(d_low_cong, d_high_cong)

        # Case B: Carried delay > 2.0 hours (forfeited slot, congestion penalty applies)
        # For carried_delay = 6.0 h, excess = 4.0 h. Penalty rate = 0.35 h/hr.
        # Congestion 0.8: penalty = 4.0 * 0.35 * 0.8 = 1.12 h -> Total = 17.12 h
        # Congestion 0.2: penalty = 4.0 * 0.35 * 0.2 = 0.28 h -> Total = 16.28 h
        d_above_high = simulate_single_berth_duration(
            sched_berth_hours=16.0,
            carried_delay_at_anchorage=6.0,
            port_congestion_index=0.8,
            noise_hours=0.0,
        )
        d_above_low = simulate_single_berth_duration(
            sched_berth_hours=16.0,
            carried_delay_at_anchorage=6.0,
            port_congestion_index=0.2,
            noise_hours=0.0,
        )
        self.assertAlmostEqual(d_above_high, 17.12, places=2)
        self.assertAlmostEqual(d_above_low, 16.28, places=2)
        self.assertGreater(d_above_high, d_above_low)

    def test_nonnegative_durations(self):
        for r in self.records:
            self.assertGreaterEqual(r["departure_delay_hours"], 0.0)
            self.assertGreater(r["actual_sea_hours"], 0.0)
            self.assertGreater(r["actual_berth_hours"], 0.0)
            self.assertGreater(r["actual_terminal_hours"], 0.0)
            self.assertGreater(r["actual_customs_hours"], 0.0)
            self.assertGreater(r["actual_drayage_hours"], 0.0)
            self.assertGreater(r["remaining_duration_hours"], 0.0)

    def test_timestamp_continuity_and_sequence(self):
        for r in self.records:
            # Parse scheduled sequence
            t_s0 = datetime.fromisoformat(r["scheduled_departure_time"])
            t_s1 = datetime.fromisoformat(r["scheduled_anchorage_arrival"])
            t_s2 = datetime.fromisoformat(r["scheduled_berth_time"])
            t_s3 = datetime.fromisoformat(r["scheduled_terminal_end"])
            t_s4 = datetime.fromisoformat(r["scheduled_customs_end"])
            t_s5 = datetime.fromisoformat(r["scheduled_final_delivery_time"])

            # Assert strict monotonicity of scheduled sequence
            self.assertTrue(t_s0 < t_s1 < t_s2 < t_s3 < t_s4 < t_s5)

            # Parse actual sequence
            t_a0 = datetime.fromisoformat(r["actual_departure_time"])
            t_a1 = datetime.fromisoformat(r["actual_anchorage_arrival"])
            t_a2 = datetime.fromisoformat(r["actual_berth_time"])
            t_a3 = datetime.fromisoformat(r["actual_terminal_end"])
            t_a4 = datetime.fromisoformat(r["actual_customs_end"])
            t_a5 = datetime.fromisoformat(r["actual_final_delivery_time"])

            # Assert strict monotonicity of actual realized sequence
            self.assertTrue(t_a0 < t_a1 < t_a2 < t_a3 < t_a4 < t_a5)

            # Check actual departure = scheduled departure + departure delay
            diff_dep = (t_a0 - t_s0).total_seconds() / 3600.0
            self.assertAlmostEqual(diff_dep, r["departure_delay_hours"], places=2)

            # Check individual actual stage durations match timestamp steps (within 0.05 h)
            self.assertAlmostEqual((t_a1 - t_a0).total_seconds() / 3600.0, r["actual_sea_hours"], places=1)
            self.assertAlmostEqual((t_a2 - t_a1).total_seconds() / 3600.0, r["actual_berth_hours"], places=1)
            self.assertAlmostEqual((t_a3 - t_a2).total_seconds() / 3600.0, r["actual_terminal_hours"], places=1)
            self.assertAlmostEqual((t_a4 - t_a3).total_seconds() / 3600.0, r["actual_customs_hours"], places=1)
            self.assertAlmostEqual((t_a5 - t_a4).total_seconds() / 3600.0, r["actual_drayage_hours"], places=1)

    def test_target_closure(self):
        for r in self.records:
            # y_rem = d_actual_3 + d_actual_4 + d_actual_5
            expected_rem = r["actual_terminal_hours"] + r["actual_customs_hours"] + r["actual_drayage_hours"]
            self.assertAlmostEqual(r["remaining_duration_hours"], expected_rem, places=2)

            # y_delay = (t_actual_5 - t_sched_5) in hours
            t_s5 = datetime.fromisoformat(r["scheduled_final_delivery_time"])
            t_a5 = datetime.fromisoformat(r["actual_final_delivery_time"])
            expected_delay = (t_a5 - t_s5).total_seconds() / 3600.0
            self.assertAlmostEqual(r["final_delay_hours"], expected_delay, places=2)

            # delay_risk_binary == 1 if final_delay_hours > 6.0 else 0
            expected_risk = 1 if r["final_delay_hours"] > DELAY_RISK_THRESHOLD_HOURS else 0
            self.assertEqual(r["delay_risk_binary"], expected_risk)

    def test_feature_whitelist_and_leakage_assertion(self):
        # 1. Whitelist has no overlap with prohibited blacklist
        whitelist_set = set(FEATURE_WHITELIST)
        blacklist_set = set(PROHIBITED_LEAKAGE_COLUMNS)
        overlap = whitelist_set.intersection(blacklist_set)
        self.assertEqual(len(overlap), 0, f"Found prohibited columns in feature whitelist: {overlap}")

        # 2. All whitelist columns exist in generated records
        all_keys = set(self.records[0].keys())
        missing_whitelist = whitelist_set - all_keys
        self.assertEqual(len(missing_whitelist), 0, f"Whitelist columns missing from record keys: {missing_whitelist}")

    def test_chronological_split_integrity(self):
        split_idx = int(NUM_SHIPMENTS * TRAIN_RATIO)
        train_records = self.records[:split_idx]
        test_records = self.records[split_idx:]

        self.assertEqual(len(train_records), 200)
        self.assertEqual(len(test_records), 50)

        # Ensure strict chronological ordering by scheduled departure time
        max_train_dep = max(datetime.fromisoformat(r["scheduled_departure_time"]) for r in train_records)
        min_test_dep = min(datetime.fromisoformat(r["scheduled_departure_time"]) for r in test_records)
        self.assertLess(max_train_dep, min_test_dep)

        # Ensure all 5 corridors are represented in both train and test sets
        train_corridors = set(r["route_id"] for r in train_records)
        test_corridors = set(r["route_id"] for r in test_records)
        expected_corridors = set(CORRIDORS.keys())

        self.assertEqual(train_corridors, expected_corridors)
        self.assertEqual(test_corridors, expected_corridors)


if __name__ == "__main__":
    unittest.main()
