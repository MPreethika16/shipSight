"""Baseline and ML modeling pipeline for ShipSight.
Implements the scheduled-duration baseline and scikit-learn HistGradientBoostingRegressor,
evaluates on a strictly chronological test set, and reports MAE, RMSE, and derived delay risk.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    DELAY_RISK_THRESHOLD_HOURS,
    FEATURE_WHITELIST,
    PROHIBITED_LEAKAGE_COLUMNS,
    TRAIN_RATIO,
)


def load_dataset(csv_path: str = os.path.join("data", "synthetic_shipments.csv")) -> List[Dict[str, Any]]:
    """Loads CSV dataset and parses numeric/string fields."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generator first.")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    # Cast numeric fields
    for r in records:
        for k, v in r.items():
            if k in [
                "scheduled_sea_hours",
                "scheduled_berth_hours",
                "scheduled_terminal_hours",
                "scheduled_customs_hours",
                "scheduled_drayage_hours",
                "departure_delay_hours",
                "weather_severity_index",
                "transit_distance_nm",
                "actual_sea_hours",
                "carried_delay_at_anchorage",
                "port_congestion_index",
                "actual_berth_hours",
                "carried_delay_at_berth",
                "actual_terminal_hours",
                "actual_customs_hours",
                "actual_drayage_hours",
                "remaining_duration_hours",
                "final_delay_hours",
            ]:
                r[k] = float(v)
            elif k in ["container_teu", "customs_inspection_flag", "delay_risk_binary"]:
                r[k] = int(v)

    # Ensure strictly sorted chronologically by scheduled departure
    records.sort(key=lambda x: x["scheduled_departure_time"])
    return records


def split_chronological(
    records: List[Dict[str, Any]],
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records into chronological train and test partitions."""
    split_idx = int(len(records) * train_ratio)
    train_set = records[:split_idx]
    test_set = records[split_idx:]
    return train_set, test_set


def build_feature_matrix(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features X (strictly whitelisted), remaining duration y_rem,

    final delay y_delay, and binary risk labels.
    """
    X_dicts: List[Dict[str, Any]] = []
    y_rem_list: List[float] = []
    y_delay_list: List[float] = []
    y_risk_list: List[int] = []

    for r in records:
        # Leakage guard: assert zero prohibited columns entered
        for prohibited in PROHIBITED_LEAKAGE_COLUMNS:
            if prohibited in FEATURE_WHITELIST:
                raise ValueError(f"CRITICAL LEAKAGE: Prohibited column {prohibited} found in FEATURE_WHITELIST!")

        # Whitelist extraction
        feat_dict = {col: r[col] for col in FEATURE_WHITELIST}
        X_dicts.append(feat_dict)
        y_rem_list.append(r["remaining_duration_hours"])
        y_delay_list.append(r["final_delay_hours"])
        y_risk_list.append(r["delay_risk_binary"])

    return (
        X_dicts,
        np.array(y_rem_list, dtype=np.float64),
        np.array(y_delay_list, dtype=np.float64),
        np.array(y_risk_list, dtype=np.int32),
    )


def build_pipeline() -> Pipeline:
    """Builds scikit-learn pipeline with dense OneHotEncoder and HistGradientBoostingRegressor."""
    categorical_cols = ["route_id", "carrier_code", "cargo_type"]
    numerical_cols = [col for col in FEATURE_WHITELIST if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ],
        remainder="drop",
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=4,
        min_samples_leaf=10,
        learning_rate=0.08,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def calculate_scheduled_remaining_duration(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Calculates the scheduled downstream allowance (sum of scheduled terminal, customs, and drayage hours)."""
    return np.array(
        [
            float(r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"])
            for r in records
        ],
        dtype=np.float64,
    )


class ResidualETAPredictor:
    """Estimator that fits on residual delay (actual_remaining - scheduled_remaining)
    and reconstructs remaining duration as max(0.0, scheduled_remaining + predicted_residual).
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline if pipeline is not None else build_pipeline()
        self.is_residual = True

    def fit(
        self,
        X: pd.DataFrame,
        y_actual_rem: np.ndarray,
        y_sched_rem: np.ndarray,
    ) -> ResidualETAPredictor:
        y_residual = np.array(y_actual_rem, dtype=np.float64) - np.array(y_sched_rem, dtype=np.float64)
        self.pipeline.fit(X, y_residual)
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self.pipeline.predict(X), dtype=np.float64)

    def predict(
        self,
        X: pd.DataFrame,
        y_sched_rem: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        pred_res = self.predict_residual(X)
        if y_sched_rem is None:
            if isinstance(X, pd.DataFrame) and all(
                c in X.columns for c in ["scheduled_terminal_hours", "scheduled_customs_hours", "scheduled_drayage_hours"]
            ):
                sched_arr = (
                    X["scheduled_terminal_hours"] + X["scheduled_customs_hours"] + X["scheduled_drayage_hours"]
                ).to_numpy(dtype=np.float64)
            else:
                raise ValueError("y_sched_rem must be provided if X does not contain scheduled stage hours.")
        else:
            sched_arr = np.array(y_sched_rem, dtype=np.float64)
        return np.maximum(0.0, sched_arr + pred_res)



def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    records: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    """Calculates regression (MAE, RMSE, MAPE) and derived delay-risk classification metrics."""
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    # Documented limitation: MAPE can be noisy when target is near-zero, but included with caveat
    mape = float(np.mean(np.abs(errors / np.clip(y_true, 1e-3, None))) * 100.0)

    # Derived predicted final delay:
    # y_pred_delay = carried_delay_at_berth + (y_pred_rem - scheduled_downstream_hours)
    pred_delays = []
    for i, r in enumerate(records):
        sched_downstream = r["scheduled_terminal_hours"] + r["scheduled_customs_hours"] + r["scheduled_drayage_hours"]
        pred_delay = r["carried_delay_at_berth"] + (y_pred[i] - sched_downstream)
        pred_delays.append(pred_delay)

    pred_delays_arr = np.array(pred_delays, dtype=np.float64)
    pred_risk = (pred_delays_arr > DELAY_RISK_THRESHOLD_HOURS).astype(int)
    true_risk = np.array([r["delay_risk_binary"] for r in records], dtype=int)

    prec = float(precision_score(true_risk, pred_risk, zero_division=0))
    rec = float(recall_score(true_risk, pred_risk, zero_division=0))
    f1 = float(f1_score(true_risk, pred_risk, zero_division=0))
    cm = confusion_matrix(true_risk, pred_risk, labels=[0, 1]).tolist()

    return {
        "model_name": model_name,
        "n_eval": len(y_true),
        "mae_hours": round(mae, 3),
        "rmse_hours": round(rmse, 3),
        "mape_pct": round(mape, 2),
        "delay_risk_threshold_hours": DELAY_RISK_THRESHOLD_HOURS,
        "risk_precision": round(prec, 3),
        "risk_recall": round(rec, 3),
        "risk_f1": round(f1, 3),
        "confusion_matrix": cm,
    }


def run_training_and_evaluation() -> Dict[str, Any]:
    """Executes the complete training, evaluation, and feature attribution pipeline."""
    records = load_dataset()
    train_records, test_records = split_chronological(records)

    X_train_dicts, y_train_rem, _, _ = build_feature_matrix(train_records)
    X_test_dicts, y_test_rem, _, _ = build_feature_matrix(test_records)

    y_train_sched = calculate_scheduled_remaining_duration(train_records)
    y_test_sched = calculate_scheduled_remaining_duration(test_records)

    # 1. Baseline Model: Scheduled downstream duration
    baseline_metrics = compute_metrics(y_test_rem, y_test_sched, test_records, "Scheduled Duration Baseline")

    # 2. Scikit-learn HistGradientBoostingRegressor Pipeline (Direct Absolute Target)
    X_train_df = pd.DataFrame(X_train_dicts)
    X_test_df = pd.DataFrame(X_test_dicts)

    pipeline_direct = build_pipeline()
    pipeline_direct.fit(X_train_df, y_train_rem)
    y_test_pred_direct = pipeline_direct.predict(X_test_df)

    direct_metrics = compute_metrics(y_test_rem, y_test_pred_direct, test_records, "HistGradientBoosting (Direct)")

    # 3. Residual Learning Predictor: y_residual = y_rem - y_sched
    residual_predictor = ResidualETAPredictor()
    residual_predictor.fit(X_train_df, y_train_rem, y_train_sched)
    y_test_pred_residual = residual_predictor.predict(X_test_df, y_test_sched)

    residual_metrics = compute_metrics(
        y_test_rem, y_test_pred_residual, test_records, "HistGradientBoosting (Residual)"
    )

    # 4. Permutation Feature Importance on held-out test set (Direct Pipeline)
    perm_result = permutation_importance(
        pipeline_direct,
        X_test_df,
        y_test_rem,
        n_repeats=15,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    feature_importances: List[Dict[str, Any]] = []
    for col_idx, col_name in enumerate(FEATURE_WHITELIST):
        imp_mean = float(perm_result.importances_mean[col_idx])
        imp_std = float(perm_result.importances_std[col_idx])
        feature_importances.append({"feature": col_name, "mean_mae_increase": round(imp_mean, 3), "std": round(imp_std, 3)})

    # Sort descending by importance
    feature_importances.sort(key=lambda x: x["mean_mae_increase"], reverse=True)

    return {
        "train_size": len(train_records),
        "test_size": len(test_records),
        "baseline": baseline_metrics,
        "hist_gradient_boosting": direct_metrics,
        "hist_gradient_boosting_residual": residual_metrics,
        "model": direct_metrics,
        "residual_model": residual_metrics,
        "feature_importances": feature_importances,
        "pipeline": pipeline_direct,
        "pipeline_direct": pipeline_direct,
        "pipeline_residual": residual_predictor,
        "residual_predictor": residual_predictor,
    }


if __name__ == "__main__":
    results = run_training_and_evaluation()
    print("=== MODEL EVALUATION RESULTS ===")
    print(f"Train records: {results['train_size']}, Test records: {results['test_size']}")
    print("\n[Baseline Model]")
    print(f"  MAE: {results['baseline']['mae_hours']} hrs | RMSE: {results['baseline']['rmse_hours']} hrs | F1: {results['baseline']['risk_f1']}")
    print("\n[Direct HistGradientBoostingRegressor]")
    print(f"  MAE: {results['hist_gradient_boosting']['mae_hours']} hrs | RMSE: {results['hist_gradient_boosting']['rmse_hours']} hrs | F1: {results['hist_gradient_boosting']['risk_f1']}")
    print("\n[Residual HistGradientBoostingRegressor]")
    print(f"  MAE: {results['hist_gradient_boosting_residual']['mae_hours']} hrs | RMSE: {results['hist_gradient_boosting_residual']['rmse_hours']} hrs | F1: {results['hist_gradient_boosting_residual']['risk_f1']}")
    print("\nTop 5 Feature Attributions (Direct Model MAE Degradation When Shuffled):")
    for f in results["feature_importances"][:5]:
        print(f"  - {f['feature']}: +{f['mean_mae_increase']} hrs")
