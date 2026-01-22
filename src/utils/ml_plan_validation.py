"""
ML Plan Validation - Coherence checks for Plan ↔ Code ↔ Data consistency.

SENIOR REASONING PATTERN:
This module validates that the generated code implements the ml_plan correctly.
It checks coherence between:
- data_profile.json (Evidence)
- ml_plan.json (Decision Plan)
- Generated ML code (Execution)
"""

import re
from typing import Dict, Any, List, Tuple


def validate_plan_code_coherence(
    ml_plan: Dict[str, Any],
    code: str,
    data_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Validate that code implements the ml_plan correctly.

    Args:
        ml_plan: The ml_plan.json dict
        code: The generated Python code
        data_profile: Optional data_profile.json for additional checks

    Returns:
        {
            "passed": bool,
            "status": "APPROVED" | "REJECTED",
            "violations": List[str],  # List of specific violations
            "warnings": List[str],    # Non-fatal issues
            "feedback": str,          # Human-readable feedback
        }
    """
    violations: List[str] = []
    warnings: List[str] = []

    if not ml_plan:
        warnings.append("ml_plan is empty; cannot validate coherence")
        return {
            "passed": True,
            "status": "APPROVED",
            "violations": [],
            "warnings": warnings,
            "feedback": "No ml_plan to validate against.",
        }

    if not code or len(code.strip()) < 50:
        violations.append("Code is empty or too short")
        return {
            "passed": False,
            "status": "REJECTED",
            "violations": violations,
            "warnings": warnings,
            "feedback": "PLAN_CODE_VIOLATION: Code is empty or too short.",
        }

    code_lower = code.lower()

    # 1. Check training_rows_policy coherence
    training_rows_policy = ml_plan.get("training_rows_policy", "")

    if training_rows_policy == "only_rows_with_label":
        # Code must filter rows where y is not missing
        # Look for patterns like: y.notna(), y.dropna(), ~y.isna(), df[df[target].notna()]
        filter_patterns = [
            r"\.notna\(\)",
            r"\.dropna\(",
            r"~.*\.isna\(\)",
            r"\[.*notna\(",
            r"train_mask\s*=",
            r"\.notnull\(\)",
            r"~.*\.isnull\(\)",
        ]
        has_filter = any(re.search(pat, code) for pat in filter_patterns)
        if not has_filter:
            violations.append(
                f"PLAN_VIOLATION:training_rows_policy='{training_rows_policy}' "
                "but code does not filter rows by label availability "
                "(expected patterns: .notna(), .dropna(), ~.isna())"
            )

    elif training_rows_policy == "use_split_column":
        split_column = ml_plan.get("split_column")
        if split_column:
            # Code must reference the split column
            if split_column.lower() not in code_lower:
                # Try variations
                col_variations = [
                    split_column.lower(),
                    split_column.lower().replace("_", ""),
                    split_column.lower().replace("-", ""),
                ]
                found = any(v in code_lower for v in col_variations)
                if not found:
                    violations.append(
                        f"PLAN_VIOLATION:training_rows_policy='use_split_column' "
                        f"with split_column='{split_column}' but code does not reference it"
                    )

    # 2. Check metric_policy coherence
    metric_policy = ml_plan.get("metric_policy", {})
    primary_metric = metric_policy.get("primary_metric", "")

    if primary_metric:
        # Check that the primary metric is mentioned in code
        metric_variations = {
            "roc_auc": ["roc_auc", "auc", "roc_auc_score"],
            "accuracy": ["accuracy", "accuracy_score"],
            "r2": ["r2", "r2_score", "r_squared"],
            "f1": ["f1", "f1_score"],
            "silhouette": ["silhouette", "silhouette_score"],
            "neg_mean_squared_error": ["mse", "mean_squared_error", "neg_mean_squared"],
        }
        variations = metric_variations.get(primary_metric.lower(), [primary_metric.lower()])
        found_metric = any(v in code_lower for v in variations)
        if not found_metric:
            warnings.append(
                f"PLAN_WARNING:metric_policy specifies '{primary_metric}' "
                "but metric name not found in code"
            )

    # 3. Check cv_policy coherence
    cv_policy = ml_plan.get("cv_policy", {})
    cv_strategy = cv_policy.get("strategy", "")

    if cv_strategy:
        cv_mapping = {
            "StratifiedKFold": ["stratifiedkfold", "stratified"],
            "KFold": ["kfold", "k-fold"],
            "TimeSeriesSplit": ["timeseriessplit", "time_series"],
            "GroupKFold": ["groupkfold", "group"],
        }
        expected_patterns = cv_mapping.get(cv_strategy, [cv_strategy.lower()])
        found_cv = any(pat in code_lower for pat in expected_patterns)
        # Also check for cross_val_score which implies CV is used
        has_cross_val = "cross_val" in code_lower or "cv=" in code_lower
        if not found_cv and not has_cross_val:
            warnings.append(
                f"PLAN_WARNING:cv_policy specifies '{cv_strategy}' "
                "but CV strategy not clearly found in code"
            )

    # 4. Check leakage_policy coherence
    leakage_policy = ml_plan.get("leakage_policy", {})
    flagged_columns = leakage_policy.get("flagged_columns", [])

    for flagged_col in flagged_columns[:5]:  # Check first 5
        if flagged_col and flagged_col.lower() in code_lower:
            # Check if it's used as a feature (not just printed/logged)
            # Pattern: features = [..., 'flagged_col', ...] or X[['flagged_col', ...]]
            feature_pattern = rf"['\"]({re.escape(flagged_col)})['\"]"
            if re.search(feature_pattern, code):
                # Check if it's in a feature list context (rough heuristic)
                # This is a soft warning since we can't perfectly detect usage
                warnings.append(
                    f"PLAN_WARNING:leakage_policy flagged '{flagged_col}' "
                    "which appears in code - verify it's not used as feature"
                )

    # Build final result
    passed = len(violations) == 0
    status = "APPROVED" if passed else "REJECTED"

    feedback_parts = []
    if violations:
        feedback_parts.append("PLAN_CODE_VIOLATIONS:")
        feedback_parts.extend([f"  - {v}" for v in violations])
    if warnings:
        feedback_parts.append("WARNINGS:")
        feedback_parts.extend([f"  - {w}" for w in warnings])
    if passed and not warnings:
        feedback_parts.append("Plan-Code coherence check passed.")

    return {
        "passed": passed,
        "status": status,
        "violations": violations,
        "warnings": warnings,
        "feedback": "\n".join(feedback_parts),
    }


def validate_plan_data_coherence(
    ml_plan: Dict[str, Any],
    data_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that ml_plan is consistent with data_profile evidence.

    Args:
        ml_plan: The ml_plan.json dict
        data_profile: The data_profile.json dict

    Returns:
        {
            "passed": bool,
            "status": "APPROVED" | "REJECTED",
            "inconsistencies": List[str],
            "feedback": str,
        }
    """
    inconsistencies: List[str] = []

    if not ml_plan or not data_profile:
        return {
            "passed": True,
            "status": "APPROVED",
            "inconsistencies": [],
            "feedback": "No plan or profile to validate.",
        }

    # 1. Check if training_rows_policy matches outcome analysis
    training_rows_policy = ml_plan.get("training_rows_policy", "")
    outcome_analysis = data_profile.get("outcome_analysis", {})

    if training_rows_policy == "only_rows_with_label":
        # There should be outcome missingness evidence
        has_missing_outcome = False
        for col, analysis in outcome_analysis.items():
            if isinstance(analysis, dict):
                null_frac = analysis.get("null_frac", 0)
                if null_frac and null_frac > 0.01:
                    has_missing_outcome = True
                    break
        if not has_missing_outcome:
            inconsistencies.append(
                f"ml_plan says training_rows_policy='only_rows_with_label' "
                "but data_profile shows no significant outcome missingness"
            )

    elif training_rows_policy == "use_split_column":
        # There should be split candidates
        split_candidates = data_profile.get("split_candidates", [])
        split_column = ml_plan.get("split_column")
        if split_column and split_candidates:
            found = any(
                c.get("column") == split_column
                for c in split_candidates
                if isinstance(c, dict)
            )
            if not found:
                inconsistencies.append(
                    f"ml_plan says use split_column='{split_column}' "
                    "but this column is not in data_profile.split_candidates"
                )
        elif not split_candidates:
            inconsistencies.append(
                "ml_plan says 'use_split_column' "
                "but data_profile has no split_candidates"
            )

    passed = len(inconsistencies) == 0
    status = "APPROVED" if passed else "REJECTED"

    feedback = ""
    if inconsistencies:
        feedback = "PLAN_DATA_INCONSISTENCIES:\n" + "\n".join(
            [f"  - {i}" for i in inconsistencies]
        )
    else:
        feedback = "Plan-Data coherence check passed."

    return {
        "passed": passed,
        "status": status,
        "inconsistencies": inconsistencies,
        "feedback": feedback,
    }


def run_full_coherence_validation(
    ml_plan: Dict[str, Any],
    code: str,
    data_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run full coherence validation: Plan ↔ Code and Plan ↔ Data.

    Returns combined result with all checks.
    """
    plan_code_result = validate_plan_code_coherence(ml_plan, code, data_profile)
    plan_data_result = {"passed": True, "status": "APPROVED", "inconsistencies": [], "feedback": ""}

    if data_profile:
        plan_data_result = validate_plan_data_coherence(ml_plan, data_profile)

    # Combine results
    all_violations = plan_code_result.get("violations", [])
    all_warnings = plan_code_result.get("warnings", []) + plan_data_result.get("inconsistencies", [])

    passed = plan_code_result.get("passed", True) and plan_data_result.get("passed", True)
    status = "APPROVED" if passed else "REJECTED"

    feedback_parts = []
    if plan_code_result.get("feedback"):
        feedback_parts.append(plan_code_result["feedback"])
    if plan_data_result.get("feedback") and not plan_data_result.get("passed"):
        feedback_parts.append(plan_data_result["feedback"])

    return {
        "passed": passed,
        "status": status,
        "violations": all_violations,
        "warnings": all_warnings,
        "plan_code_check": plan_code_result,
        "plan_data_check": plan_data_result,
        "feedback": "\n\n".join(feedback_parts) if feedback_parts else "All coherence checks passed.",
    }
