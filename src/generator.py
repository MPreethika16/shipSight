"""Deterministic synthetic data generator for ShipSight.
Models multi-stage shipment journeys with sequence-consistent timestamps,
explicit delay-propagation equations, schedule buffers, and strict mathematical closure.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np

from src.config import (
    CARGO_TYPES,
    CARRIERS,
    CORRIDORS,
    DELAY_RISK_THRESHOLD_HOURS,
    NUM_SHIPMENTS,
    RANDOM_SEED,
)


def simulate_single_berth_duration(
    sched_berth_hours: float,
    carried_delay_at_anchorage: float,
    port_congestion_index: float,
    noise_hours: float = 0.0,
) -> float:
    """Deterministic formula for actual anchorage and berthing duration.
    
    If carried_delay_at_anchorage <= 2.0 hours, the vessel arrives within its reserved
    pilotage/berth buffer, incurring zero congestion penalty.
    If carried_delay_at_anchorage > 2.0 hours, the vessel forfeits its window and
    incurs a congestion queue penalty proportional to port congestion.
    """
    excess_delay = max(0.0, carried_delay_at_anchorage - 2.0)
    congestion_penalty = 0.35 * excess_delay * port_congestion_index
    return float(sched_berth_hours + congestion_penalty + noise_hours)


def generate_synthetic_shipments(
    num_shipments: int = NUM_SHIPMENTS,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Generates synthetic shipment records with mathematically consistent sequential timestamps."""
    rng = np.random.default_rng(seed=seed)

    records: List[Dict[str, Any]] = []
    base_start_time = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    corridor_keys = list(CORRIDORS.keys())

    for i in range(num_shipments):
        shipment_id = f"SHP-{i + 1:03d}"
        route_id = corridor_keys[i % len(corridor_keys)]
        corridor = CORRIDORS[route_id]

        carrier_code = str(rng.choice(CARRIERS))
        cargo_type = str(rng.choice(CARGO_TYPES, p=[0.70, 0.20, 0.10]))
        container_teu = int(rng.choice([1, 2, 4], p=[0.40, 0.45, 0.15]))

        # Spread scheduled departures across days chronologically
        departure_offset_hours = (i // len(corridor_keys)) * 14.0 + (i % len(corridor_keys)) * 2.5
        t_sched_0 = base_start_time + timedelta(hours=departure_offset_hours)

        # Scheduled Stage Durations (Hours)
        d_sched_1 = float(corridor["distance_nm"] / corridor["nominal_speed_kts"])
        d_sched_2 = float(corridor["sched_berth_hours"])
        d_sched_3 = float(corridor["sched_terminal_hours"])
        d_sched_4 = float(corridor["sched_customs_hours"])
        d_sched_5 = float(corridor["sched_drayage_hours"])

        # Sequential Scheduled Baseline Timestamps
        t_sched_1 = t_sched_0 + timedelta(hours=d_sched_1)
        t_sched_2 = t_sched_1 + timedelta(hours=d_sched_2)
        t_sched_3 = t_sched_2 + timedelta(hours=d_sched_3)
        t_sched_4 = t_sched_3 + timedelta(hours=d_sched_4)
        t_sched_5 = t_sched_4 + timedelta(hours=d_sched_5)

        # Stage 0: Origin Departure (45% on-time, 55% delayed by up to 24h)
        dep_delay = float(np.clip(rng.exponential(scale=4.0), 0.0, 24.0)) if rng.random() < 0.55 else 0.0
        t_actual_0 = t_sched_0 + timedelta(hours=dep_delay)

        # Stage 1: Maritime Sea Transit
        # Scheduled duration is nominal at mean weather W = 2.3.
        # Favorable weather (W < 2.3) allows slight speed gain; adverse weather (W > 2.3) causes transit drag.
        weather_index = float(np.clip(rng.normal(loc=2.3, scale=0.8), 1.0, 5.0))
        transit_distance = float(corridor["distance_nm"])
        sea_factor = 1.0 + 0.035 * (weather_index - 2.3)
        sea_noise = float(rng.normal(loc=0.0, scale=3.5))
        d_actual_1 = float(max(0.85 * d_sched_1, d_sched_1 * sea_factor + sea_noise))
        t_actual_1 = t_actual_0 + timedelta(hours=d_actual_1)

        # Anchorage arrival delay (D_1)
        carried_delay_anchorage = float((t_actual_1 - t_sched_1).total_seconds() / 3600.0)

        # Stage 2: Port Anchorage & Berthing (Threshold equation)
        port_congestion = float(np.clip(rng.beta(a=2.0, b=3.0), 0.0, 1.0))
        berth_noise = float(rng.normal(loc=0.0, scale=2.5))
        d_actual_2 = max(
            0.6 * d_sched_2,
            simulate_single_berth_duration(
                sched_berth_hours=d_sched_2,
                carried_delay_at_anchorage=carried_delay_anchorage,
                port_congestion_index=port_congestion,
                noise_hours=berth_noise,
            ),
        )
        t_actual_2 = t_actual_1 + timedelta(hours=d_actual_2)  # PREDICTION CUTOFF t_pred

        # Carried delay at berthing (D_2)
        carried_delay_berth = float((t_actual_2 - t_sched_2).total_seconds() / 3600.0)

        # Manifest flag known at t_pred (Hazmat and Reefer have higher inspection propensity)
        inspection_prob = 0.45 if cargo_type == "Hazmat" else (0.25 if cargo_type == "Reefer" else 0.12)
        customs_inspection_flag = int(rng.random() < inspection_prob)

        # Stage 3: Terminal Handling (Partial buffer absorption)
        term_noise = float(rng.normal(loc=0.0, scale=3.0))
        term_adjustment = 0.15 * max(0.0, carried_delay_berth) - 4.0 if carried_delay_berth > 0 else -2.5
        d_actual_3 = float(max(0.7 * d_sched_3, d_sched_3 + term_adjustment + term_noise))
        t_actual_3 = t_actual_2 + timedelta(hours=d_actual_3)

        carried_delay_term = float((t_actual_3 - t_sched_3).total_seconds() / 3600.0)

        # Stage 4: Customs & Regulatory Clearance
        # Physical hold adds 24-48h; standard automated electronic clearance clears on time or slightly early
        inspect_hold = float(rng.uniform(low=24.0, high=48.0)) if customs_inspection_flag else float(rng.normal(loc=-1.0, scale=2.0))
        d_actual_4 = float(max(0.5 * d_sched_4, d_sched_4 + inspect_hold))
        t_actual_4 = t_actual_3 + timedelta(hours=d_actual_4)

        carried_delay_customs = float((t_actual_4 - t_sched_4).total_seconds() / 3600.0)

        # Stage 5: Inland Drayage
        dray_noise = float(rng.normal(loc=0.0, scale=2.5))
        dray_compounding = 0.10 * max(0.0, carried_delay_customs)
        d_actual_5 = float(max(0.6 * d_sched_5, d_sched_5 + dray_compounding + dray_noise))
        t_actual_5 = t_actual_4 + timedelta(hours=d_actual_5)

        # Round stage durations to 2 decimal places to guarantee exact arithmetic closure
        d_actual_1 = round(d_actual_1, 2)
        d_actual_2 = round(d_actual_2, 2)
        d_actual_3 = round(d_actual_3, 2)
        d_actual_4 = round(d_actual_4, 2)
        d_actual_5 = round(d_actual_5, 2)

        # Ground Truth Targets derived directly from exact rounded stage durations
        y_rem = round(d_actual_3 + d_actual_4 + d_actual_5, 2)
        y_delay = round((t_actual_5 - t_sched_5).total_seconds() / 3600.0, 2)
        y_risk = int(y_delay > DELAY_RISK_THRESHOLD_HOURS)

        record: Dict[str, Any] = {
            # Metadata
            "shipment_id": shipment_id,
            "route_id": route_id,
            "carrier_code": carrier_code,
            "cargo_type": cargo_type,
            "container_teu": container_teu,
            # Scheduled Baseline
            "scheduled_departure_time": t_sched_0.isoformat(),
            "scheduled_sea_hours": round(d_sched_1, 2),
            "scheduled_anchorage_arrival": t_sched_1.isoformat(),
            "scheduled_berth_hours": round(d_sched_2, 2),
            "scheduled_berth_time": t_sched_2.isoformat(),
            "scheduled_terminal_hours": round(d_sched_3, 2),
            "scheduled_terminal_end": t_sched_3.isoformat(),
            "scheduled_customs_hours": round(d_sched_4, 2),
            "scheduled_customs_end": t_sched_4.isoformat(),
            "scheduled_drayage_hours": round(d_sched_5, 2),
            "scheduled_final_delivery_time": t_sched_5.isoformat(),
            # Upstream Realized Metrics Known at t_pred
            "departure_delay_hours": round(dep_delay, 2),
            "actual_departure_time": t_actual_0.isoformat(),
            "weather_severity_index": round(weather_index, 2),
            "transit_distance_nm": round(transit_distance, 1),
            "actual_sea_hours": d_actual_1,
            "actual_anchorage_arrival": t_actual_1.isoformat(),
            "carried_delay_at_anchorage": round(carried_delay_anchorage, 2),
            "port_congestion_index": round(port_congestion, 3),
            "actual_berth_hours": d_actual_2,
            "actual_berth_time": t_actual_2.isoformat(),  # PREDICTION CUTOFF
            "carried_delay_at_berth": round(carried_delay_berth, 2),
            "customs_inspection_flag": customs_inspection_flag,
            # Downstream Realized Metrics (PROHIBITED LEAKAGE)
            "actual_terminal_hours": d_actual_3,
            "actual_terminal_end": t_actual_3.isoformat(),
            "actual_customs_hours": d_actual_4,
            "actual_customs_end": t_actual_4.isoformat(),
            "actual_drayage_hours": d_actual_5,
            "actual_final_delivery_time": t_actual_5.isoformat(),
            # Targets
            "remaining_duration_hours": y_rem,
            "final_delay_hours": y_delay,
            "delay_risk_binary": y_risk,
        }
        records.append(record)

    # Sort strictly chronologically by scheduled departure time
    records.sort(key=lambda r: r["scheduled_departure_time"])
    return records


def export_synthetic_csv(
    records: List[Dict[str, Any]],
    output_path: str = os.path.join("data", "synthetic_shipments.csv"),
) -> str:
    """Exports generated records to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not records:
        raise ValueError("Cannot export empty records list.")

    headers = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    return output_path


if __name__ == "__main__":
    recs = generate_synthetic_shipments()
    path = export_synthetic_csv(recs)
    print(f"Generated {len(recs)} synthetic shipment records saved to: {path}")
