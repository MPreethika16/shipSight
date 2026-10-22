"""Delay propagation and explainability analysis for ShipSight.
Calculates stage-to-stage delay transmission coefficients, buffer absorption rates,
and correlation across the shipment journey lifecycle.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import numpy as np

from src.model import load_dataset


def analyze_delay_propagation(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes how delays propagate across the 5 sequential stages.
    
    Stages tracked:
      D0: Departure Delay (Origin)
      D1: Carried Delay at Anchorage Arrival
      D2: Carried Delay at Berthing (t_pred cutoff)
      D3: Carried Delay at Terminal Exit
      D4: Carried Delay at Customs Release
      D5: Final Delivery Delay
    """
    d0_vals = np.array([r["departure_delay_hours"] for r in records], dtype=np.float64)
    d1_vals = np.array([r["carried_delay_at_anchorage"] for r in records], dtype=np.float64)
    d2_vals = np.array([r["carried_delay_at_berth"] for r in records], dtype=np.float64)

    # Reconstruct D3, D4, D5
    d3_vals = []
    d4_vals = []
    d5_vals = []
    for r in records:
        # D_k = T_actual_k - T_sched_k in hours
        # D_3 = D_2 + (actual_terminal - sched_terminal)
        d3 = r["carried_delay_at_berth"] + (r["actual_terminal_hours"] - r["scheduled_terminal_hours"])
        d4 = d3 + (r["actual_customs_hours"] - r["scheduled_customs_hours"])
        d5 = r["final_delay_hours"]
        d3_vals.append(d3)
        d4_vals.append(d4)
        d5_vals.append(d5)

    d3_arr = np.array(d3_vals, dtype=np.float64)
    d4_arr = np.array(d4_vals, dtype=np.float64)
    d5_arr = np.array(d5_vals, dtype=np.float64)

    # 1. Pearson Correlation Matrix across cumulative stage delays
    all_stages = np.vstack([d0_vals, d1_vals, d2_vals, d3_arr, d4_arr, d5_arr])
    corr_matrix = np.corrcoef(all_stages)

    # 2. Stage-to-stage transmission slope (OLS regression slope: D_{k} ~ alpha * D_{k-1})
    def calc_transmission_slope(x: np.ndarray, y: np.ndarray) -> float:
        denom = float(np.sum((x - np.mean(x)) ** 2))
        if denom == 0.0:
            return 0.0
        return float(np.sum((x - np.mean(x)) * (y - np.mean(y))) / denom)

    slope_0_to_1 = calc_transmission_slope(d0_vals, d1_vals)
    slope_1_to_2 = calc_transmission_slope(d1_vals, d2_vals)
    slope_2_to_3 = calc_transmission_slope(d2_vals, d3_arr)
    slope_3_to_4 = calc_transmission_slope(d3_arr, d4_arr)
    slope_4_to_5 = calc_transmission_slope(d4_arr, d5_arr)

    # 3. Buffer absorption evaluation at terminal
    # Compare shipments with high vs low arrival delay at berth
    mild_berth_delay = [r for r, d2 in zip(records, d2_vals) if d2 <= 4.0]
    severe_berth_delay = [r for r, d2 in zip(records, d2_vals) if d2 > 4.0]

    terminal_dev_mild = np.mean([r["actual_terminal_hours"] - r["scheduled_terminal_hours"] for r in mild_berth_delay])
    terminal_dev_severe = np.mean([r["actual_terminal_hours"] - r["scheduled_terminal_hours"] for r in severe_berth_delay])

    # 4. Customs inspection impact
    inspected = [r for r in records if r["customs_inspection_flag"] == 1]
    cleared_auto = [r for r in records if r["customs_inspection_flag"] == 0]

    mean_customs_inspected = float(np.mean([r["actual_customs_hours"] for r in inspected]))
    mean_customs_auto = float(np.mean([r["actual_customs_hours"] for r in cleared_auto]))

    stage_names = ["D0: Departure", "D1: Anchorage", "D2: Berthing", "D3: Terminal", "D4: Customs", "D5: Final Delivery"]

    return {
        "num_records": len(records),
        "stage_names": stage_names,
        "correlation_matrix": [[round(val, 3) for val in row] for row in corr_matrix],
        "transmission_slopes": {
            "Stage 0 -> Stage 1 (Departure to Anchorage)": round(slope_0_to_1, 3),
            "Stage 1 -> Stage 2 (Anchorage to Berthing)": round(slope_1_to_2, 3),
            "Stage 2 -> Stage 3 (Berthing to Terminal)": round(slope_2_to_3, 3),
            "Stage 3 -> Stage 4 (Terminal to Customs)": round(slope_3_to_4, 3),
            "Stage 4 -> Stage 5 (Customs to Delivery)": round(slope_4_to_5, 3),
        },
        "buffer_analysis": {
            "mild_berth_delay_count": len(mild_berth_delay),
            "severe_berth_delay_count": len(severe_berth_delay),
            "avg_terminal_schedule_deviation_mild_hrs": round(float(terminal_dev_mild), 2),
            "avg_terminal_schedule_deviation_severe_hrs": round(float(terminal_dev_severe), 2),
        },
        "customs_impact": {
            "inspected_count": len(inspected),
            "auto_cleared_count": len(cleared_auto),
            "avg_customs_hours_inspected": round(mean_customs_inspected, 2),
            "avg_customs_hours_auto_cleared": round(mean_customs_auto, 2),
            "added_delay_from_inspection_hrs": round(mean_customs_inspected - mean_customs_auto, 2),
        },
    }


if __name__ == "__main__":
    records = load_dataset()
    results = analyze_delay_propagation(records)
    print("=== STAGE-BY-STAGE DELAY PROPAGATION ANALYSIS ===")
    print(f"Total shipments analyzed: {results['num_records']}")
    print("\nStage-to-Stage Transmission Slopes (OLS Delta D_k / Delta D_{k-1}):")
    for transition, slope in results["transmission_slopes"].items():
        print(f"  {transition}: {slope}")

    print("\nTerminal Yard Buffer Absorption Analysis:")
    print(f"  Mild Berth Delay (<= 4h) Count: {results['buffer_analysis']['mild_berth_delay_count']}")
    print(f"  Mild Berth Avg Terminal Deviation: {results['buffer_analysis']['avg_terminal_schedule_deviation_mild_hrs']} hrs")
    print(f"  Severe Berth Delay (> 4h) Count: {results['buffer_analysis']['severe_berth_delay_count']}")
    print(f"  Severe Berth Avg Terminal Deviation: {results['buffer_analysis']['avg_terminal_schedule_deviation_severe_hrs']} hrs")

    print("\nCustoms Clearance Disruption Analysis:")
    print(f"  Inspected Count: {results['customs_impact']['inspected_count']}, Mean Dwell: {results['customs_impact']['avg_customs_hours_inspected']} hrs")
    print(f"  Auto Cleared Count: {results['customs_impact']['auto_cleared_count']}, Mean Dwell: {results['customs_impact']['avg_customs_hours_auto_cleared']} hrs")
    print(f"  Net Disruption Impact: +{results['customs_impact']['added_delay_from_inspection_hrs']} hrs")
