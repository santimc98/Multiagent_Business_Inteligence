import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_target_column(contract: Dict[str, Any], df: pd.DataFrame | None) -> str | None:
    reqs = contract.get("data_requirements", []) or []
    target_candidates = []
    for req in reqs:
        role = str(req.get("role", "")).lower()
        if role in {"derived_label", "target_label"}:
            name = req.get("canonical_name") or req.get("name")
            if name:
                target_candidates.append(str(name))
    for candidate in target_candidates:
        if df is not None and candidate in df.columns:
            return candidate
    return target_candidates[0] if target_candidates else None


def _calc_lift(baseline: float | None, model: float | None, higher_is_better: bool) -> float | None:
    if baseline is None or model is None:
        return None
    if higher_is_better:
        return model - baseline
    if baseline == 0:
        return None
    return (baseline - model) / baseline


def _segment_coverage(case_summary: pd.DataFrame | None, min_size: int | None) -> Tuple[float | None, int]:
    if case_summary is None or min_size is None or "Segment_Size" not in case_summary.columns:
        return None, 0
    try:
        small = case_summary["Segment_Size"].astype(float) < float(min_size)
        if small.empty:
            return None, 0
        return float(small.mean()), int(small.sum())
    except Exception:
        return None, 0


def build_data_adequacy_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    weights = _safe_load_json("data/weights.json")
    cleaned = _safe_load_csv("data/cleaned_data.csv")
    case_summary = _safe_load_csv("data/case_summary.csv")

    cls_metrics = weights.get("classification_metrics", {}) if isinstance(weights, dict) else {}
    reg_metrics = weights.get("regression_metrics", {}) if isinstance(weights, dict) else {}

    f1 = cls_metrics.get("f1_score_cv_mean")
    f1_baseline = cls_metrics.get("baseline_f1")
    mae = reg_metrics.get("mae_cv_mean")
    mae_baseline = reg_metrics.get("baseline_mae")

    f1_lift = _calc_lift(f1_baseline, f1, higher_is_better=True)
    mae_lift = _calc_lift(mae_baseline, mae, higher_is_better=False)

    row_count = int(cleaned.shape[0]) if cleaned is not None else None
    feature_count = None
    if isinstance(weights, dict):
        feat = weights.get("feature_importance")
        if isinstance(feat, dict):
            feature_count = len(feat)

    rows_per_feature = None
    if row_count and feature_count:
        rows_per_feature = row_count / max(1, feature_count)

    target_col = _find_target_column(contract, cleaned)
    class_balance = None
    if cleaned is not None and target_col and target_col in cleaned.columns:
        try:
            class_balance = float(cleaned[target_col].mean())
        except Exception:
            class_balance = None

    quality_gates = contract.get("quality_gates", {}) if isinstance(contract, dict) else {}
    min_segment_size = quality_gates.get("min_segment_size")
    small_segment_frac, small_segment_count = _segment_coverage(case_summary, min_segment_size)

    reasons: List[str] = []
    signals: Dict[str, Any] = {
        "row_count": row_count,
        "feature_count": feature_count,
        "rows_per_feature": rows_per_feature,
        "class_balance": class_balance,
        "small_segment_fraction": small_segment_frac,
        "small_segment_count": small_segment_count,
        "f1_score_cv_mean": f1,
        "baseline_f1": f1_baseline,
        "f1_lift": f1_lift,
        "mae_cv_mean": mae,
        "baseline_mae": mae_baseline,
        "mae_lift": mae_lift,
    }

    if f1_lift is not None and f1_lift < 0.05:
        reasons.append("classification_lift_low")
    if mae_lift is not None and mae_lift < 0.1:
        reasons.append("regression_lift_low")
    if rows_per_feature is not None and rows_per_feature < 10:
        reasons.append("high_dimensionality_low_sample")
    if class_balance is not None and (class_balance < 0.1 or class_balance > 0.9):
        reasons.append("class_imbalance")
    if small_segment_frac is not None and small_segment_frac > 0.3:
        reasons.append("segments_too_small")

    data_limited = len(reasons) >= 2 or (
        (f1_lift is not None and f1_lift < 0.02) and (mae_lift is not None and mae_lift < 0.05)
    )

    recommendations: List[str] = []
    if "classification_lift_low" in reasons:
        recommendations.append("Collect more labeled outcomes or refine the success label definition.")
    if "regression_lift_low" in reasons:
        recommendations.append("Increase the number of successful contracts with reliable 1stYearAmount values.")
    if "high_dimensionality_low_sample" in reasons:
        recommendations.append("Increase sample size or reduce feature dimensionality through aggregation.")
    if "class_imbalance" in reasons:
        recommendations.append("Improve class balance by collecting more rare outcomes or sampling evenly.")
    if "segments_too_small" in reasons:
        recommendations.append("Aggregate segments or collect more cases per segment before recommending prices.")

    status = "data_limited" if data_limited else "sufficient_signal"
    threshold = int(state.get("data_adequacy_threshold", 3) or 3)
    consecutive = int(state.get("data_adequacy_consecutive", 0) or 0)

    return {
        "status": status,
        "reasons": reasons,
        "recommendations": recommendations,
        "signals": signals,
        "consecutive_data_limited": consecutive,
        "data_limited_threshold": threshold,
        "threshold_reached": consecutive >= threshold,
    }


def write_data_adequacy_report(state: Dict[str, Any], path: str = "data/data_adequacy_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = build_data_adequacy_report(state)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
