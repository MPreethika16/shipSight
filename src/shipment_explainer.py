"""Shipment-Level ETA and Delay Explanation Engine for ShipSight.

Generates transparent, auditable explanations for individual shipment predictions
at the vessel berthing cutoff (t_pred = actual_berth_time).

Deconstructs upstream observed delays, downstream model estimates,
and risk classification, strictly distinguishing model estimates from
simulation parameters.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import DELAY_RISK_THRESHOLD_HOURS, FEATURE_WHITELIST


def explain_shipment(
    record: Dict[str, Any],
    pipeline: Pipeline,
) -> Dict[str, Any]:
    """Generates an auditable explanation for an individual shipment prediction.

    Parameters:
        record: A single shipment dictionary from the dataset.
        pipeline: Trained scikit-learn pipeline (HistGradientBoostingRegressor).

    Returns:
        A structured dictionary containing timestamps, observed upstream delays,
        model-derived downstream predictions, and delay-risk classification.
    """
    shipment_id = record.get("shipment_id", "UNKNOWN")
    route_id = record["route_id"]

    # 1. Prediction Cutoff and Whitelisted Input Features
    feat_dict = {col: record[col] for col in FEATURE_WHITELIST}
    X_single = pd.DataFrame([feat_dict])

    # 2. Model Prediction (Remaining Duration from Berth to Delivery)
    y_pred_rem_raw = float(pipeline.predict(X_single)[0])
    y_pred_rem = round(y_pred_rem_raw, 2)

    # 3. Timestamps
    actual_berth_time = datetime.fromisoformat(record["actual_berth_time"])
    scheduled_final_delivery = datetime.fromisoformat(record["scheduled_final_delivery_time"])
    predicted_final_eta = actual_berth_time + timedelta(hours=y_pred_rem)

    # 4. Scheduled Downstream Buffer
    sched_downstream_hours = float(
        record["scheduled_terminal_hours"]
        + record["scheduled_customs_hours"]
        + record["scheduled_drayage_hours"]
    )

    # 5. Delay Deconstruction
    carried_delay_at_berth = float(record["carried_delay_at_berth"])
    predicted_downstream_delta = float(y_pred_rem - sched_downstream_hours)
    predicted_final_delay_hours = float(carried_delay_at_berth + predicted_downstream_delta)
    predicted_risk_binary = int(predicted_final_delay_hours > DELAY_RISK_THRESHOLD_HOURS)

    # Actual ground truth for comparison if present
    actual_final_delay = record.get("final_delay_hours")
    actual_final_delivery = record.get("actual_final_delivery_time")

    explanation = {
        "shipment_id": shipment_id,
        "route_id": route_id,
        "carrier_code": record["carrier_code"],
        "cargo_type": record["cargo_type"],
        "container_teu": record["container_teu"],
        "prediction_cutoff": {
            "milestone": "Vessel Berthing (t_pred)",
            "timestamp": actual_berth_time.isoformat(),
        },
        "timestamps": {
            "scheduled_final_delivery": scheduled_final_delivery.isoformat(),
            "predicted_final_eta": predicted_final_eta.isoformat(),
            "actual_final_delivery": actual_final_delivery,
        },
        "durations_and_delays": {
            "carried_delay_at_berth_hours": round(carried_delay_at_berth, 2),
            "scheduled_downstream_allowance_hours": round(sched_downstream_hours, 2),
            "predicted_remaining_duration_hours": round(y_pred_rem, 2),
            "predicted_downstream_delta_hours": round(predicted_downstream_delta, 2),
            "predicted_final_delay_hours": round(predicted_final_delay_hours, 2),
            "actual_final_delay_hours": round(actual_final_delay, 2) if actual_final_delay is not None else None,
            "delay_error_hours": round(predicted_final_delay_hours - actual_final_delay, 2)
            if actual_final_delay is not None
            else None,
        },
        "delay_risk": {
            "threshold_hours": DELAY_RISK_THRESHOLD_HOURS,
            "predicted_risk": "DELAYED" if predicted_risk_binary == 1 else "ON-TIME",
            "predicted_risk_binary": predicted_risk_binary,
            "actual_risk_binary": record.get("delay_risk_binary"),
        },
        "observed_upstream_stages": [
            {
                "stage": "Stage 0: Origin Departure",
                "departure_delay_hours": round(record.get("departure_delay_hours", 0.0), 2),
            },
            {
                "stage": "Stage 1: Sea Transit",
                "transit_distance_nm": record["transit_distance_nm"],
                "weather_severity_index": round(record["weather_severity_index"], 2),
                "scheduled_sea_hours": round(record.get("scheduled_sea_hours", 0.0), 2),
                "actual_sea_hours": round(record.get("actual_sea_hours", 0.0), 2),
                "carried_delay_at_anchorage_hours": round(record["carried_delay_at_anchorage"], 2),
            },
            {
                "stage": "Stage 2: Anchorage & Berthing",
                "port_congestion_index": round(record["port_congestion_index"], 3),
                "scheduled_berth_hours": round(record.get("scheduled_berth_hours", 0.0), 2),
                "actual_berth_hours": round(record.get("actual_berth_hours", 0.0), 2),
                "carried_delay_at_berth_hours": round(record["carried_delay_at_berth"], 2),
            },
        ],
        "planned_downstream_stages": [
            {
                "stage": "Stage 3: Container Terminal Handling",
                "scheduled_hours": record["scheduled_terminal_hours"],
            },
            {
                "stage": "Stage 4: Customs Clearance",
                "scheduled_hours": record["scheduled_customs_hours"],
                "customs_inspection_flag": record["customs_inspection_flag"],
                "inspection_note": "Advance administrative hold flag known at berth; physical inspection expected"
                if record["customs_inspection_flag"] == 1
                else "Automated electronic clearance expected",
            },
            {
                "stage": "Stage 5: Inland Drayage",
                "scheduled_hours": record["scheduled_drayage_hours"],
            },
        ],
        "methodology_disclaimer": (
            "Model-derived estimates reflect GBDT remaining-duration predictions based on the whitelisted feature set. "
            "Stage details reflect synthetic simulation parameters and must not be interpreted as measured real-world causal effects."
        ),
    }

    return explanation


def format_shipment_explanation(exp: Dict[str, Any]) -> str:
    """Formats an explanation dictionary into a clean, human-readable terminal report."""
    d = exp["durations_and_delays"]
    t = exp["timestamps"]
    r = exp["delay_risk"]

    lines = [
        "=" * 72,
        f"SHIPSIGHT INDIVIDUAL SHIPMENT ETA EXPLANATION: {exp['shipment_id']}",
        "=" * 72,
        f"Corridor: {exp['route_id']} | Carrier: {exp['carrier_code']} | Cargo: {exp['cargo_type']} ({exp['container_teu']} TEU)",
        f"Prediction Cutoff: {exp['prediction_cutoff']['milestone']} @ {exp['prediction_cutoff']['timestamp']}",
        "-" * 72,
        "TIMESTAMPS & ETA FORECAST:",
        f"  * Scheduled Final Delivery : {t['scheduled_final_delivery']}",
        f"  * Predicted Final ETA      : {t['predicted_final_eta']}",
    ]
    if t["actual_final_delivery"]:
        lines.append(f"  * Actual Final Delivery    : {t['actual_final_delivery']}")

    lines.extend(
        [
            "-" * 72,
            "DELAY DECONSTRUCTION (HOURS):",
            f"  1. Observed Upstream Delay @ Berth   : {d['carried_delay_at_berth_hours']:>+6.2f} hrs",
            f"  2. Scheduled Downstream Allowance    :  {d['scheduled_downstream_allowance_hours']:>5.2f} hrs",
            f"  3. Model-Predicted Remaining Duration:  {d['predicted_remaining_duration_hours']:>5.2f} hrs",
            f"  4. Predicted Downstream Slippage     : {d['predicted_downstream_delta_hours']:>+6.2f} hrs",
            f"  => Total Predicted Final Delay       : {d['predicted_final_delay_hours']:>+6.2f} hrs",
        ]
    )
    if d["actual_final_delay_hours"] is not None:
        lines.append(
            f"  (Ground-Truth Final Delay: {d['actual_final_delay_hours']:>+6.2f} hrs | Forecast Error: {d['delay_error_hours']:>+6.2f} hrs)"
        )

    lines.extend(
        [
            "-" * 72,
            f"OPERATIONAL DELAY-RISK CLASSIFICATION (Threshold = {r['threshold_hours']} hrs):",
            f"  * Risk Assessment: {r['predicted_risk']}",
        ]
    )

    lines.extend(
        [
            "-" * 72,
            "UPSTREAM REALIZED FACTORS (Known at Prediction Time):",
        ]
    )
    for stg in exp["observed_upstream_stages"]:
        lines.append(f"  * {stg['stage']}: {stg}")

    lines.extend(
        [
            "-" * 72,
            "DOWNSTREAM PLANNED BUFFERS & ADVANCE INDICATORS:",
        ]
    )
    for stg in exp["planned_downstream_stages"]:
        lines.append(f"  * {stg['stage']}: {stg}")

    lines.extend(
        [
            "-" * 72,
            f"DISCLAIMER: {exp['methodology_disclaimer']}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)
