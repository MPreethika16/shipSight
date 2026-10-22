"""Shipment-Level ETA and Delay Explanation Engine for ShipSight.

Generates transparent, auditable explanations for individual shipment predictions
at the vessel berthing cutoff (t_pred = actual_berth_time).

Provides two distinct operational modes:
1. 'live': Operational decision-support mode. Strictly enforces pre-cutoff data
   quarantine. Never accesses or reveals realized downstream delivery milestones,
   actual delays, or ground-truth risk.
2. 'retrospective': Post-voyage performance evaluation mode. Audits forecast accuracy
   and risk classification against realized final delivery outcomes and dwell times.

Includes scenario sensitivity calculations by evaluating the actual fitted model
pipeline on modified copies of admissible features (e.g. customs hold flag, port congestion).
All what-if sensitivities are explicitly documented as model sensitivities, not causal attributions.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import DELAY_RISK_THRESHOLD_HOURS, FEATURE_WHITELIST


def calculate_scenario_sensitivity(
    feat_dict: Dict[str, Any],
    pipeline: Pipeline,
    actual_berth_time: datetime,
    sched_downstream_hours: float,
    carried_delay_at_berth: float,
    y_pred_rem_baseline: float,
) -> Dict[str, Any]:
    """Evaluates the fitted model pipeline on perturbed admissible features.

    Parameters:
        feat_dict: Whitelisted admissible feature dictionary at prediction cutoff.
        pipeline: Trained scikit-learn pipeline.
        actual_berth_time: Timestamp of vessel berthing cutoff.
        sched_downstream_hours: Scheduled downstream allowance from berth to delivery.
        carried_delay_at_berth: Observed upstream delay accumulated up to berthing.
        y_pred_rem_baseline: Baseline model prediction for remaining duration.

    Returns:
        Structured scenario sensitivity dictionary comparing baseline predictions
        with counterfactual model outputs. Explicitly labeled as model sensitivity.
    """
    baseline_final_delay = carried_delay_at_berth + (y_pred_rem_baseline - sched_downstream_hours)
    baseline_eta = actual_berth_time + timedelta(hours=y_pred_rem_baseline)
    baseline_risk = "DELAYED" if baseline_final_delay > DELAY_RISK_THRESHOLD_HOURS else "ON-TIME"

    scenarios: List[Dict[str, Any]] = []

    # Scenario 1: customs_inspection_flag toggle (Administrative hold vs automated clearance)
    if "customs_inspection_flag" in feat_dict:
        orig_flag = int(feat_dict["customs_inspection_flag"])
        counterfactual_flag = 1 if orig_flag == 0 else 0

        feat_scen1 = copy.deepcopy(feat_dict)
        feat_scen1["customs_inspection_flag"] = counterfactual_flag

        X_scen1 = pd.DataFrame([feat_scen1])
        scen1_rem_raw = float(pipeline.predict(X_scen1)[0])
        scen1_rem = round(scen1_rem_raw, 2)
        scen1_delta_rem = round(scen1_rem - y_pred_rem_baseline, 2)
        scen1_final_delay = round(carried_delay_at_berth + (scen1_rem - sched_downstream_hours), 2)
        scen1_eta = actual_berth_time + timedelta(hours=scen1_rem)
        scen1_risk = "DELAYED" if scen1_final_delay > DELAY_RISK_THRESHOLD_HOURS else "ON-TIME"

        scenarios.append(
            {
                "scenario_name": "toggle_customs_inspection_flag",
                "feature_modified": "customs_inspection_flag",
                "baseline_value": orig_flag,
                "scenario_value": counterfactual_flag,
                "description": (
                    f"Counterfactual administrative clearance flag: toggled from {orig_flag} "
                    f"({'hold/inspection' if orig_flag == 1 else 'auto-clearance'}) to {counterfactual_flag} "
                    f"({'hold/inspection' if counterfactual_flag == 1 else 'auto-clearance'})."
                ),
                "scenario_predicted_remaining_hours": scen1_rem,
                "scenario_predicted_final_delay_hours": scen1_final_delay,
                "scenario_predicted_final_eta": scen1_eta.isoformat(),
                "scenario_predicted_risk": scen1_risk,
                "sensitivity_remaining_delta_hours": scen1_delta_rem,
                "sensitivity_final_delay_delta_hours": scen1_delta_rem,
            }
        )

    # Scenario 2: Port congestion sensitivity (+0.30 surge if moderate/low, or -0.30 relief if high)
    if "port_congestion_index" in feat_dict:
        orig_cong = float(feat_dict["port_congestion_index"])
        if orig_cong < 0.70:
            target_cong = round(min(1.0, orig_cong + 0.30), 3)
            cong_desc = f"Simulates elevated port congestion (+0.30 surge: {orig_cong:.2f} -> {target_cong:.2f})."
            cong_name = "port_congestion_surge_plus_0_30"
        else:
            target_cong = round(max(0.0, orig_cong - 0.30), 3)
            cong_desc = f"Simulates port congestion easing (-0.30 relief: {orig_cong:.2f} -> {target_cong:.2f})."
            cong_name = "port_congestion_relief_minus_0_30"

        feat_scen2 = copy.deepcopy(feat_dict)
        feat_scen2["port_congestion_index"] = target_cong

        X_scen2 = pd.DataFrame([feat_scen2])
        scen2_rem_raw = float(pipeline.predict(X_scen2)[0])
        scen2_rem = round(scen2_rem_raw, 2)
        scen2_delta_rem = round(scen2_rem - y_pred_rem_baseline, 2)
        scen2_final_delay = round(carried_delay_at_berth + (scen2_rem - sched_downstream_hours), 2)
        scen2_eta = actual_berth_time + timedelta(hours=scen2_rem)
        scen2_risk = "DELAYED" if scen2_final_delay > DELAY_RISK_THRESHOLD_HOURS else "ON-TIME"

        scenarios.append(
            {
                "scenario_name": cong_name,
                "feature_modified": "port_congestion_index",
                "baseline_value": round(orig_cong, 3),
                "scenario_value": target_cong,
                "description": cong_desc,
                "scenario_predicted_remaining_hours": scen2_rem,
                "scenario_predicted_final_delay_hours": scen2_final_delay,
                "scenario_predicted_final_eta": scen2_eta.isoformat(),
                "scenario_predicted_risk": scen2_risk,
                "sensitivity_remaining_delta_hours": scen2_delta_rem,
                "sensitivity_final_delay_delta_hours": scen2_delta_rem,
            }
        )

    return {
        "label": "Model Sensitivity (What-If Feature Perturbation) - Not Causal Attribution",
        "methodological_classification": "Fitted Model Response Surface Perturbation",
        "interpretation_warning": (
            "Scenario predictions are generated by running the actual fitted regression pipeline on "
            "modified feature vectors. These results represent statistical model sensitivity to input variations. "
            "They are strictly NOT physical causal attributions, NOT guarantees of real-world outcome changes, "
            "and MUST NOT be interpreted as additive SHAP feature contributions."
        ),
        "baseline_prediction": {
            "predicted_remaining_duration_hours": round(y_pred_rem_baseline, 2),
            "predicted_final_delay_hours": round(baseline_final_delay, 2),
            "predicted_final_eta": baseline_eta.isoformat(),
            "predicted_risk": baseline_risk,
        },
        "scenarios": scenarios,
    }


def explain_shipment(
    record: Dict[str, Any],
    pipeline: Pipeline,
    mode: str = "live",
    evaluate_scenarios: bool = True,
) -> Dict[str, Any]:
    """Generates an auditable explanation for an individual shipment prediction.

    Parameters:
        record: A single shipment record dictionary.
        pipeline: Trained scikit-learn pipeline (HistGradientBoostingRegressor).
        mode: Operational mode. Either 'live' (strict pre-cutoff data quarantine)
              or 'retrospective' (includes ground-truth actuals and forecast error).
        evaluate_scenarios: If True, evaluates scenario sensitivity on modified
                            feature vectors using the fitted pipeline.

    Returns:
        Structured dictionary containing timestamps, observed upstream delay,
        scheduled allowance, predicted remaining duration, predicted final ETA,
        predicted delay relative to schedule, and scenario sensitivity.
    """
    if mode not in ("live", "retrospective"):
        raise ValueError(
            f"Invalid mode '{mode}'. Operational mode must be either 'live' or 'retrospective'."
        )

    shipment_id = record.get("shipment_id", "UNKNOWN")
    route_id = record["route_id"]

    # 1. Whitelisted Input Features (Admissible at Vessel Berthing Cutoff)
    feat_dict = {col: record[col] for col in FEATURE_WHITELIST}
    X_single = pd.DataFrame([feat_dict])

    # 2. Model Prediction (Remaining Duration from Berth to Delivery)
    y_pred_rem_raw = float(pipeline.predict(X_single)[0])
    y_pred_rem = round(y_pred_rem_raw, 2)

    # 3. Timestamps Known at Cutoff
    actual_berth_time = datetime.fromisoformat(record["actual_berth_time"])
    scheduled_final_delivery = datetime.fromisoformat(record["scheduled_final_delivery_time"])
    predicted_final_eta = actual_berth_time + timedelta(hours=y_pred_rem)

    # 4. Scheduled Downstream Remaining Allowance
    sched_downstream_hours = float(
        record["scheduled_terminal_hours"]
        + record["scheduled_customs_hours"]
        + record["scheduled_drayage_hours"]
    )

    # 5. Delay Deconstruction Relative to Scheduled Delivery
    carried_delay_at_berth = float(record["carried_delay_at_berth"])
    predicted_downstream_delta = float(y_pred_rem - sched_downstream_hours)
    predicted_final_delay_hours = float(carried_delay_at_berth + predicted_downstream_delta)
    predicted_risk_binary = int(predicted_final_delay_hours > DELAY_RISK_THRESHOLD_HOURS)
    predicted_risk_label = "DELAYED" if predicted_risk_binary == 1 else "ON-TIME"

    # Assemble base explanation (strict pre-cutoff operational information)
    explanation: Dict[str, Any] = {
        "mode": mode,
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
        },
        "durations_and_delays": {
            "observed_delay_at_cutoff_hours": round(carried_delay_at_berth, 2),
            "carried_delay_at_berth_hours": round(carried_delay_at_berth, 2),
            "scheduled_remaining_duration_hours": round(sched_downstream_hours, 2),
            "scheduled_downstream_allowance_hours": round(sched_downstream_hours, 2),
            "predicted_remaining_duration_hours": round(y_pred_rem, 2),
            "predicted_downstream_delta_hours": round(predicted_downstream_delta, 2),
            "predicted_final_delay_hours": round(predicted_final_delay_hours, 2),
            "predicted_delay_relative_to_schedule_hours": round(predicted_final_delay_hours, 2),
        },
        "delay_risk": {
            "threshold_hours": DELAY_RISK_THRESHOLD_HOURS,
            "predicted_risk": predicted_risk_label,
            "predicted_risk_binary": predicted_risk_binary,
        },
        "observed_upstream_stages": [
            {
                "stage": "Stage 0: Origin Departure",
                "departure_delay_hours": round(float(record.get("departure_delay_hours", 0.0)), 2),
            },
            {
                "stage": "Stage 1: Sea Transit",
                "transit_distance_nm": float(record["transit_distance_nm"]),
                "scheduled_sea_hours": round(float(record.get("scheduled_sea_hours", 0.0)), 2),
                "actual_sea_hours": round(float(record.get("actual_sea_hours", 0.0)), 2),
                "weather_severity_index": round(float(record["weather_severity_index"]), 2),
                "carried_delay_at_anchorage_hours": round(float(record["carried_delay_at_anchorage"]), 2),
            },
            {
                "stage": "Stage 2: Anchorage & Berthing",
                "port_congestion_index": round(float(record["port_congestion_index"]), 3),
                "scheduled_berth_hours": round(float(record.get("scheduled_berth_hours", 0.0)), 2),
                "actual_berth_hours": round(float(record.get("actual_berth_hours", 0.0)), 2),
                "carried_delay_at_berth_hours": round(float(record["carried_delay_at_berth"]), 2),
            },
        ],
        "planned_downstream_stages": [
            {
                "stage": "Stage 3: Container Terminal Handling",
                "scheduled_hours": round(float(record["scheduled_terminal_hours"]), 2),
            },
            {
                "stage": "Stage 4: Customs Clearance",
                "scheduled_hours": round(float(record["scheduled_customs_hours"]), 2),
                "customs_inspection_flag": int(record["customs_inspection_flag"]),
            },
            {
                "stage": "Stage 5: Inland Drayage",
                "scheduled_hours": round(float(record["scheduled_drayage_hours"]), 2),
            },
        ],
        "methodology_disclaimer": (
            "Model-derived estimates reflect GBDT remaining-duration predictions based on the whitelisted feature set. "
            "Stage details reflect synthetic simulation parameters and must not be interpreted as measured real-world causal effects."
        ),
    }

    # Scenario Sensitivity Calculation
    if evaluate_scenarios:
        explanation["scenario_sensitivity"] = calculate_scenario_sensitivity(
            feat_dict=feat_dict,
            pipeline=pipeline,
            actual_berth_time=actual_berth_time,
            sched_downstream_hours=sched_downstream_hours,
            carried_delay_at_berth=carried_delay_at_berth,
            y_pred_rem_baseline=y_pred_rem,
        )

    # Retrospective Evaluation Mode: Attach ground truth and prediction error
    if mode == "retrospective":
        actual_final_delay = record.get("final_delay_hours")
        actual_final_delivery = record.get("actual_final_delivery_time")
        actual_risk_binary = record.get("delay_risk_binary")

        if actual_final_delay is None or actual_final_delivery is None:
            raise ValueError(
                "Retrospective evaluation mode requires ground-truth outcomes ('final_delay_hours' "
                "and 'actual_final_delivery_time'), but one or more are missing from the record."
            )

        actual_rem_hours = float(
            record.get(
                "remaining_duration_hours",
                (datetime.fromisoformat(actual_final_delivery) - actual_berth_time).total_seconds() / 3600.0,
            )
        )
        rem_error_hours = round(y_pred_rem - actual_rem_hours, 2)
        delay_error_hours = round(predicted_final_delay_hours - actual_final_delay, 2)

        explanation["timestamps"]["actual_final_delivery"] = actual_final_delivery
        explanation["durations_and_delays"]["actual_final_delay_hours"] = round(actual_final_delay, 2)
        explanation["durations_and_delays"]["delay_error_hours"] = delay_error_hours
        explanation["delay_risk"]["actual_risk_binary"] = actual_risk_binary

        actual_risk_label = (
            "DELAYED" if actual_risk_binary == 1 else "ON-TIME" if actual_risk_binary == 0 else "UNKNOWN"
        )
        explanation["retrospective_evaluation"] = {
            "actual_final_delivery_time": actual_final_delivery,
            "actual_final_delay_hours": round(actual_final_delay, 2),
            "actual_remaining_duration_hours": round(actual_rem_hours, 2),
            "prediction_error_remaining_hours": rem_error_hours,
            "prediction_error_final_delay_hours": delay_error_hours,
            "actual_risk": actual_risk_label,
            "actual_risk_binary": actual_risk_binary,
            "risk_classification_correct": bool(predicted_risk_binary == actual_risk_binary)
            if actual_risk_binary is not None
            else None,
        }

    return explanation


def format_shipment_explanation(exp: Dict[str, Any]) -> str:
    """Formats an explanation dictionary into a clean, human-readable terminal report."""
    mode = exp.get("mode", "live")
    d = exp["durations_and_delays"]
    t = exp["timestamps"]
    r = exp["delay_risk"]

    mode_title = "LIVE OPERATIONAL FORECAST" if mode == "live" else "RETROSPECTIVE EVALUATION"

    lines = [
        "=" * 74,
        f"SHIPSIGHT INDIVIDUAL SHIPMENT ETA EXPLANATION [{mode_title}]",
        f"Shipment: {exp['shipment_id']} | Corridor: {exp['route_id']}",
        "=" * 74,
        f"Carrier: {exp['carrier_code']} | Cargo: {exp['cargo_type']} ({exp['container_teu']} TEU)",
        f"Prediction Cutoff : {exp['prediction_cutoff']['milestone']} @ {exp['prediction_cutoff']['timestamp']}",
        "-" * 74,
        f"{'LIVE ' if mode == 'live' else ''}TIMESTAMPS & ETA FORECAST:",
        f"  * Scheduled Final Delivery : {t['scheduled_final_delivery']}",
        f"  * Predicted Final ETA      : {t['predicted_final_eta']}",
    ]

    # In live mode, actual final delivery must NOT be revealed
    if mode == "retrospective" and "actual_final_delivery" in t and t["actual_final_delivery"]:
        lines.append(f"  * Actual Final Delivery    : {t['actual_final_delivery']}")

    lines.extend(
        [
            "-" * 74,
            "DELAY DECONSTRUCTION (HOURS):",
            f"  1. Observed Upstream Delay @ Cutoff  : {d['observed_delay_at_cutoff_hours']:>+6.2f} hrs",
            f"  2. Scheduled Remaining Duration      :  {d['scheduled_remaining_duration_hours']:>5.2f} hrs",
            f"  3. Model-Predicted Remaining Duration:  {d['predicted_remaining_duration_hours']:>5.2f} hrs",
            f"  4. Predicted Downstream Slippage     : {d['predicted_downstream_delta_hours']:>+6.2f} hrs",
            "  " + "-" * 70,
            f"  => Total Predicted Delay vs Schedule : {d['predicted_final_delay_hours']:>+6.2f} hrs",
        ]
    )

    # In live mode, ground-truth final delay and forecast error must NOT be revealed
    if mode == "retrospective" and "actual_final_delay_hours" in d and d["actual_final_delay_hours"] is not None:
        lines.extend(
            [
                "  " + "-" * 70,
                f"  * Realized Actual Final Delay        : {d['actual_final_delay_hours']:>+6.2f} hrs",
                f"  * Forecast Error (Predicted - Actual): {d['delay_error_hours']:>+6.2f} hrs",
            ]
        )

    lines.extend(
        [
            "-" * 74,
            f"OPERATIONAL DELAY-RISK CLASSIFICATION (Threshold = {r['threshold_hours']} hrs):",
            f"  * Predicted Risk Status: {r['predicted_risk']}",
        ]
    )
    if mode == "retrospective" and "retrospective_evaluation" in exp:
        retro = exp["retrospective_evaluation"]
        match_str = "CORRECT" if retro["risk_classification_correct"] else "INCORRECT"
        lines.append(f"  * Actual Realized Risk : {retro['actual_risk']} (Classification: {match_str})")

    lines.extend(
        [
            "-" * 74,
            "OBSERVED UPSTREAM STAGES (Realized at or before Cutoff):",
        ]
    )
    for stg in exp["observed_upstream_stages"]:
        lines.append(f"  * {stg['stage']}:")
        if "departure_delay_hours" in stg:
            lines.append(f"      Departure Delay        : {stg['departure_delay_hours']:>+6.2f} hrs (Observed)")
        if "actual_sea_hours" in stg:
            lines.append(
                f"      Sea Transit Duration   :  {stg['actual_sea_hours']:>5.2f} hrs "
                f"(Scheduled: {stg['scheduled_sea_hours']:.2f}h, Distance: {stg['transit_distance_nm']:.0f} nm)"
            )
            lines.append(f"      Weather Severity Index :   {stg['weather_severity_index']:>4.2f} / 5.00")
            lines.append(f"      Carried Delay @ Anch.  : {stg['carried_delay_at_anchorage_hours']:>+6.2f} hrs (Observed)")
        if "actual_berth_hours" in stg:
            lines.append(
                f"      Berth Maneuver Duration:  {stg['actual_berth_hours']:>5.2f} hrs "
                f"(Scheduled: {stg['scheduled_berth_hours']:.2f}h)"
            )
            lines.append(f"      Port Congestion Index  :  {stg['port_congestion_index']:>5.3f} / 1.000")
            lines.append(f"      Carried Delay @ Berth  : {stg['carried_delay_at_berth_hours']:>+6.2f} hrs (Cutoff D2)")

    lines.extend(
        [
            "-" * 74,
            "DOWNSTREAM PLANNED BUFFERS & ADVANCE ADMISSIBLE INDICATORS:",
        ]
    )
    for stg in exp["planned_downstream_stages"]:
        lines.append(f"  * {stg['stage']}:")
        lines.append(f"      Scheduled Allowance    :  {stg['scheduled_hours']:>5.2f} hrs (Planned Contractual)")
        if "customs_inspection_flag" in stg:
            flag_val = stg["customs_inspection_flag"]
            flag_desc = "1 (Hold / Inspection Flagged)" if flag_val == 1 else "0 (Automated Clearance Expected)"
            lines.append(f"      Advance Customs Flag   :  {flag_desc}")

    # Scenario Sensitivity Section
    if "scenario_sensitivity" in exp:
        sens = exp["scenario_sensitivity"]
        lines.extend(
            [
                "-" * 74,
                "SCENARIO SENSITIVITY (Fitted Model Re-Prediction):",
                f"[{sens['label']}]",
            ]
        )
        for scen in sens["scenarios"]:
            delta_rem = scen["sensitivity_remaining_delta_hours"]
            delta_delay = scen["sensitivity_final_delay_delta_hours"]
            lines.extend(
                [
                    f"  * Scenario: {scen['scenario_name']} ({scen['feature_modified']}: {scen['baseline_value']} -> {scen['scenario_value']})",
                    f"    - Description       : {scen['description']}",
                    f"    - Counterfactual ETA: {scen['scenario_predicted_final_eta']} "
                    f"(Remaining: {scen['scenario_predicted_remaining_hours']:.2f}h | Final Delay: {scen['scenario_predicted_final_delay_hours']:+.2f}h)",
                    f"    - Sensitivity Delta : {delta_rem:+.2f} hrs remaining duration ({delta_delay:+.2f} hrs final delay)",
                    f"    - Predicted Risk    : {scen['scenario_predicted_risk']}",
                ]
            )
        lines.append("  Note: Model sensitivity indicates response-surface shift under feature perturbation.")
        lines.append("  Never interpret as physical causal attribution or additive SHAP values.")

    lines.extend(
        [
            "-" * 74,
            f"DATA QUARANTINE & DISCLAIMER:",
            f"  * Operational Mode : {mode.upper()} ("
            + ("Strict pre-cutoff data quarantine. Zero downstream actual leakage." if mode == "live" else "Post-delivery retrospective audit against ground truth.")
            + ")",
            f"  * Disclaimer       : {exp['methodology_disclaimer']}",
            "=" * 74,
        ]
    )
    return "\n".join(lines)
