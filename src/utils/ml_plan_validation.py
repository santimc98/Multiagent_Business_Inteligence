
from typing import Dict, Any, List

def validate_ml_plan_constraints(
    plan: Dict[str, Any],
    data_profile: Dict[str, Any],
    contract: Dict[str, Any],
    strategy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate ML Plan against universal constraints.
    """
    violations = []
    
    # constraint 1: Check plan source
    source = str(plan.get("plan_source", "")).lower()
    if source.startswith("missing_") or source == "fallback":
        violations.append(f"ML_PLAN_INVALID_SOURCE: Plan source is '{source}', which indicates valid LLM generation failed.")

    # constraint 2: Outcome missingness vs training_rows_policy
    outcome_analysis = data_profile.get("outcome_analysis", {})
    has_null_labels = False
    for col, analysis in outcome_analysis.items():
        if isinstance(analysis, dict) and analysis.get("present"):
            if analysis.get("null_frac", 0) > 0:
                has_null_labels = True
                break
    
    policy = plan.get("training_rows_policy", "unspecified")
    if has_null_labels and policy == "use_all_rows":
        violations.append("ML_PLAN_CONSTRAINT_VIOLATION: Outcome has missing values, but policy is 'use_all_rows'. Must filter rows.")

    # constraint 3: Metric vs Analysis Type
    analysis_type = str(strategy.get("analysis_type", "")).lower()
    metric_policy = plan.get("metric_policy", {})
    primary_metric = str(metric_policy.get("primary_metric", "unspecified")).lower()
    
    classification_metrics = {"roc_auc", "accuracy", "f1", "precision", "recall", "log_loss", "balanced_accuracy"}
    regression_metrics = {"rmse", "mae", "r2", "mse", "mape", "rmsle"}
    
    if "classif" in analysis_type:
        if primary_metric not in classification_metrics:
            violations.append(f"ML_PLAN_METRIC_MISMATCH: Analysis is classification, but metric '{primary_metric}' is not a valid classification metric.")
    elif "regres" in analysis_type:
        if primary_metric not in regression_metrics:
            violations.append(f"ML_PLAN_METRIC_MISMATCH: Analysis is regression, but metric '{primary_metric}' is not a valid regression metric.")

    ok = len(violations) == 0
    return {
        "ok": ok,
        "violations": violations,
        "corrected_plan": plan
    }

def validate_plan_code_coherence(ml_plan: Dict[str, Any], code: str, data_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verify if the generated code respects the ML Plan policies.
    """
    violations = []
    policy = ml_plan.get("training_rows_policy", "")
    code_lower = code.lower()
    
    # 1. Label Filtering
    if policy == "only_rows_with_label":
        # Code must filter for notna() or dropna() on the target
        # Heuristic check
        if "notna()" not in code and "dropna(" not in code and "isnull()" not in code_lower:
             # Basic heuristic: if policy requires filter, code must contain filter logic
             violations.append("training_rows_policy mismatch: Plan requires 'only_rows_with_label' but code doesn't seem to filter missings (no notna/dropna)")
    
    # 2. Split Column
    if policy == "use_split_column":
        split_col = ml_plan.get("split_column", "unknown_col")
        if split_col and split_col not in code:
             violations.append(f"split_column mismatch: Plan requires using '{split_col}' but it is not referenced in code.")
             
    passed = len(violations) == 0
    return {
        "passed": passed,
        "status": "APPROVED" if passed else "REJECTED",
        "violations": violations
    }

def validate_plan_data_coherence(ml_plan: Dict[str, Any], data_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify if ML Plan is coherent with Data Profile facts.
    """
    inconsistencies = []
    
    # 1. Label Filter vs Missingness
    policy = ml_plan.get("training_rows_policy", "")
    if policy == "only_rows_with_label":
        # Check if missingness actually exists
        outcome_analysis = data_profile.get("outcome_analysis", {})
        has_nulls = False
        for col, analysis in outcome_analysis.items():
             if analysis.get("null_frac", 0) > 0:
                 has_nulls = True
        if not has_nulls:
            inconsistencies.append("Plan requires 'only_rows_with_label' but data has 0% missingness.")
            
    # 2. Split Column vs Candidates
    if policy == "use_split_column":
        split_col = ml_plan.get("split_column")
        if split_col:
             candidates = [c.get("column") for c in data_profile.get("split_candidates", [])]
             if split_col not in candidates:
                 inconsistencies.append(f"Plan uses split_column '{split_col}' but it is not in split_candidates.")

    passed = len(inconsistencies) == 0
    return {
        "passed": passed,
        "inconsistencies": inconsistencies
    }

def run_full_coherence_validation(ml_plan, code, data_profile):
    """Aggregate all checks."""
    code_res = validate_plan_code_coherence(ml_plan, code, data_profile)
    data_res = validate_plan_data_coherence(ml_plan, data_profile)
    
    violations = code_res.get("violations", [])
    warnings = data_res.get("inconsistencies", [])
    
    return {
        "passed": code_res["passed"] and data_res["passed"],
        "violations": violations,
        "warnings": warnings
    }
