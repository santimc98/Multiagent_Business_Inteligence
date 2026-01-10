"""
ML Helpers - Universal technical helpers for ML Engineer scripts.

These helpers solve common issues:
1. JSON serialization fails with numpy/int32/bool_ types
2. Sklearn scoring fails with unsupported parameters
3. Column validation against contract

DO NOT import domain-specific logic here - keep it generic!
"""

import json
import numpy as np
from typing import Any, Dict, List, Set

from src.utils.json_sanitize import dump_json, to_jsonable


def safe_json_convert(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Handles:
    - numpy.int32/int64 -> int
    - numpy.float32/float64 -> float
    - numpy.bool_ -> bool
    - numpy.ndarray -> list
    - dict keys that are not strings -> str(key)
    
    Usage:
        json.dump(safe_json_convert(data), f)
    """
    return to_jsonable(obj)


def safe_json_dump(data: Any, file_path: str, **kwargs) -> None:
    """
    Safely dump data to JSON file, converting numpy types automatically.
    
    Usage:
        safe_json_dump(metrics, 'data/metrics.json', indent=2)
    """
    dump_json(file_path, data)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely serialize data to JSON string, converting numpy types automatically.
    
    Usage:
        json_str = safe_json_dumps(metrics, indent=2)
    """
    converted = safe_json_convert(data)
    return json.dumps(converted, **kwargs)


# Sklearn-compatible scoring strings (avoid needs_proba issues)
SAFE_CLASSIFICATION_SCORERS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_weighted",
    "precision",
    "recall",
    "roc_auc",  # Works with predict_proba if available, else skipped
]

SAFE_REGRESSION_SCORERS = [
    "r2",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
]


def get_safe_scorer(task_type: str = "classification") -> str:
    """
    Return a safe default scorer string for cross_val_score.
    
    Args:
        task_type: "classification" or "regression"
        
    Returns:
        Safe scorer string that works with most models.
    """
    if task_type.lower() == "regression":
        return "neg_mean_squared_error"
    return "accuracy"


def get_available_scorers(task_type: str = "classification") -> List[str]:
    """
    Return list of safe scorer strings for the given task type.
    """
    if task_type.lower() == "regression":
        return SAFE_REGRESSION_SCORERS.copy()
    return SAFE_CLASSIFICATION_SCORERS.copy()


def validate_columns_against_contract(
    df_columns: List[str],
    contract: Dict[str, Any],
    raise_on_missing: bool = True
) -> Dict[str, List[str]]:
    """
    Validate DataFrame columns against contract-defined columns.
    
    Args:
        df_columns: List of column names from DataFrame
        contract: Execution contract dict
        raise_on_missing: If True, raise ValueError on missing required columns
        
    Returns:
        Dict with keys: 'canonical', 'derived', 'extra', 'missing'
    """
    # Get contract columns
    canonical = set(contract.get("canonical_columns", []))
    derived = set(contract.get("derived_columns", []))
    allowed = canonical | derived
    
    df_cols_set = set(df_columns)
    
    result = {
        "canonical": list(canonical & df_cols_set),
        "derived": list(derived & df_cols_set),
        "extra": list(df_cols_set - allowed),
        "missing": list(canonical - df_cols_set),
    }
    
    if raise_on_missing and result["missing"]:
        raise ValueError(f"Missing required columns from contract: {result['missing']}")
    
    return result


def validate_feature_set(
    features: List[str],
    allowed_feature_sets: Dict[str, Any],
    phase: str = "modeling"
) -> Dict[str, Any]:
    """
    Validate that features used in a phase are allowed by the contract.
    
    Args:
        features: List of feature names being used
        allowed_feature_sets: Contract's allowed_feature_sets dict
        phase: Phase name (e.g., "segmentation", "modeling", "optimization")
        
    Returns:
        Dict with 'valid': bool, 'violations': List[str], 'allowed': List[str]
    """
    if not allowed_feature_sets or not isinstance(allowed_feature_sets, dict):
        # No restrictions defined
        return {"valid": True, "violations": [], "allowed": features}
    
    # Get allowed features for this phase
    phase_key = f"{phase}_features"
    allowed = set(allowed_feature_sets.get(phase_key, []))
    
    # Also check for forbidden features
    forbidden = set(allowed_feature_sets.get("forbidden_for_modeling", []))
    forbidden |= set(allowed_feature_sets.get("audit_only_features", []))
    
    features_set = set(features)
    
    violations = []
    
    # Check if using forbidden features
    forbidden_used = features_set & forbidden
    if forbidden_used:
        violations.append(f"Using forbidden features: {list(forbidden_used)}")
    
    # If allowed set is defined, check subset
    if allowed:
        not_allowed = features_set - allowed - forbidden
        if not_allowed:
            violations.append(f"Features not in allowed set for {phase}: {list(not_allowed)}")
    
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "allowed": list(features_set - forbidden),
    }


# Code preflight patterns to detect common issues
SYNTHETIC_DATA_PATTERNS = [
    r"make_classification",
    r"make_regression",
    r"np\.random\.(rand|randn|choice|uniform|normal)\s*\(",
    r"faker\.",
    r"pd\.DataFrame\s*\(\s*\{",  # Literal DataFrame creation (suspicious)
]

HARDCODED_PATH_PATTERNS = [
    r"['\"]\/[a-z]+\/[a-z]+\.csv['\"]",  # Absolute paths
    r"['\"]C:\\\\",  # Windows absolute paths
    r"['\"]\.\.\/",  # Parent directory access
]


def preflight_check_code(
    code: str,
    contract: Dict[str, Any],
    ml_data_path: str = "data/cleaned_data.csv"
) -> Dict[str, Any]:
    """
    Preflight check for ML code to detect common contract violations.
    
    Checks:
    1. Synthetic data generation (make_classification, np.random for data creation)
    2. Hardcoded paths ignoring context
    3. Columns not in contract (basic heuristic, not foolproof)
    
    Returns:
        Dict with 'passed': bool, 'warnings': List[str], 'errors': List[str]
    """
    import re
    
    warnings = []
    errors = []
    
    # Check for synthetic data patterns
    for pattern in SYNTHETIC_DATA_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            errors.append(f"Detected synthetic data generation pattern: {pattern}")
    
    # Check for hardcoded path patterns (but allow the expected ml_data_path)
    for pattern in HARDCODED_PATH_PATTERNS:
        matches = re.findall(pattern, code, re.IGNORECASE)
        for match in matches:
            if ml_data_path not in match:
                warnings.append(f"Suspicious hardcoded path: {match}")
    
    # Basic column usage check (heuristic: look for df['ColumnName'] patterns)
    canonical = set(contract.get("canonical_columns", []))
    derived = set(contract.get("derived_columns", []))
    allowed_cols = canonical | derived
    
    # Find column access patterns
    col_pattern = r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
    used_cols = set(re.findall(col_pattern, code))
    
    # Check for columns not in contract
    unknown_cols = used_cols - allowed_cols - {"index", "Unnamed: 0"}
    if unknown_cols and allowed_cols:  # Only warn if contract has columns defined
        warnings.append(f"Columns used but not in contract: {list(unknown_cols)[:5]}")
    
    return {
        "passed": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }
