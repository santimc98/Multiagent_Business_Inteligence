"""
Feature selectors for wide datasets.

Provides compact representation of many columns via regex/prefix patterns
without losing any columns.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def expand_feature_selectors(
    df_columns: List[str],
    selectors: List[Dict[str, Any]]
) -> List[str]:
    """
    Expand feature selectors to actual column names.

    Args:
        df_columns: List of all column names in the dataframe
        selectors: List of selector dicts with type, pattern/value, role

    Returns:
        List of column names that match any selector
    """
    if not selectors:
        return []

    matched = set()

    for sel in selectors:
        sel_type = sel.get("type", "")

        if sel_type == "regex":
            pattern = sel.get("pattern", "")
            if pattern:
                try:
                    regex = re.compile(pattern)
                    for col in df_columns:
                        if regex.match(col):
                            matched.add(col)
                except re.error:
                    pass  # Invalid regex, skip

        elif sel_type == "prefix":
            prefix = sel.get("value", "")
            if prefix:
                for col in df_columns:
                    if col.startswith(prefix):
                        matched.add(col)

        elif sel_type == "suffix":
            suffix = sel.get("value", "")
            if suffix:
                for col in df_columns:
                    if col.endswith(suffix):
                        matched.add(col)

        elif sel_type == "contains":
            substring = sel.get("value", "")
            if substring:
                for col in df_columns:
                    if substring in col:
                        matched.add(col)

        elif sel_type == "list":
            # Explicit list of columns
            cols = sel.get("columns", [])
            for col in cols:
                if col in df_columns:
                    matched.add(col)

    return sorted(matched)


def infer_feature_selectors(
    df_columns: List[str],
    max_list_size: int = 200,
    min_group_size: int = 50
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Infer feature selectors from column names for wide datasets.

    Heuristics (universal, no dataset hardcode):
    1) Detect groups "prefix + number" (regex: ^(prefix)(\\d+)$)
       - If group_size >= min_group_size -> create regex selector
    2) Detect common prefixes (up to "_")
       - If group_size >= min_group_size -> create prefix selector
    3) Remaining columns -> return as explicit list

    Args:
        df_columns: List of all column names
        max_list_size: Threshold to trigger selector inference
        min_group_size: Minimum group size to create a selector

    Returns:
        (selectors, remaining_columns)
        - selectors: List of inferred selector dicts
        - remaining_columns: Columns not covered by selectors
    """
    if len(df_columns) <= max_list_size:
        # Small dataset, no need for selectors
        return [], list(df_columns)

    selectors = []
    covered = set()

    # Heuristic 1: Detect prefix+digit patterns
    # Pattern: word characters followed by digits
    prefix_digit_groups = defaultdict(list)
    prefix_digit_pattern = re.compile(r'^([a-zA-Z_]+)(\d+)$')

    for col in df_columns:
        match = prefix_digit_pattern.match(col)
        if match:
            prefix = match.group(1)
            prefix_digit_groups[prefix].append(col)

    for prefix, cols in prefix_digit_groups.items():
        if len(cols) >= min_group_size:
            # Create regex selector for this group
            selectors.append({
                "type": "regex",
                "pattern": f"^{re.escape(prefix)}\\d+$",
                "role": "model_feature",
                "description": f"Auto-inferred: {len(cols)} columns matching {prefix}N pattern",
                "count": len(cols)
            })
            covered.update(cols)

    # Heuristic 2: Detect common prefixes (before underscore)
    prefix_groups = defaultdict(list)

    for col in df_columns:
        if col in covered:
            continue
        # Find prefix before first underscore
        if "_" in col:
            prefix = col.split("_")[0] + "_"
            prefix_groups[prefix].append(col)

    for prefix, cols in prefix_groups.items():
        if len(cols) >= min_group_size:
            selectors.append({
                "type": "prefix",
                "value": prefix,
                "role": "model_feature",
                "description": f"Auto-inferred: {len(cols)} columns with prefix '{prefix}'",
                "count": len(cols)
            })
            covered.update(cols)

    # Heuristic 3: Detect common suffixes
    suffix_groups = defaultdict(list)

    for col in df_columns:
        if col in covered:
            continue
        if "_" in col:
            suffix = "_" + col.split("_")[-1]
            suffix_groups[suffix].append(col)

    for suffix, cols in suffix_groups.items():
        if len(cols) >= min_group_size:
            selectors.append({
                "type": "suffix",
                "value": suffix,
                "role": "model_feature",
                "description": f"Auto-inferred: {len(cols)} columns with suffix '{suffix}'",
                "count": len(cols)
            })
            covered.update(cols)

    # Remaining columns not covered by selectors
    remaining = [col for col in df_columns if col not in covered]

    return selectors, remaining


def get_all_feature_columns(
    df_columns: List[str],
    explicit_features: List[str],
    selectors: List[Dict[str, Any]]
) -> List[str]:
    """
    Get all feature columns combining explicit list and selectors.

    Args:
        df_columns: All columns in the dataframe
        explicit_features: Explicitly listed feature columns
        selectors: Feature selectors to expand

    Returns:
        Combined unique list of feature columns
    """
    features = set(explicit_features)
    features.update(expand_feature_selectors(df_columns, selectors))
    return sorted(features)


def compact_column_representation(
    columns: List[str],
    max_display: int = 20
) -> Dict[str, Any]:
    """
    Create a compact representation of columns for display/prompts.

    Args:
        columns: Full list of columns
        max_display: Maximum columns to list explicitly

    Returns:
        {
            "total_count": N,
            "displayed": [...first N...],
            "inferred_selectors": [...],
            "truncated": bool
        }
    """
    selectors, remaining = infer_feature_selectors(
        columns,
        max_list_size=max_display,
        min_group_size=10  # Lower threshold for display
    )

    truncated = len(remaining) > max_display

    return {
        "total_count": len(columns),
        "displayed": remaining[:max_display] if truncated else remaining,
        "inferred_selectors": selectors,
        "truncated": truncated,
        "hidden_count": max(0, len(remaining) - max_display) if truncated else 0
    }


def merge_selectors_with_explicit(
    selectors: List[Dict[str, Any]],
    explicit_columns: List[str],
    max_explicit: int = 100
) -> Dict[str, Any]:
    """
    Merge feature selectors with explicit column list.

    Returns a structure suitable for contract.allowed_feature_sets.

    Args:
        selectors: Inferred or declared selectors
        explicit_columns: Explicitly listed columns
        max_explicit: Max columns to keep in explicit list

    Returns:
        {
            "core": [...explicit up to max...],
            "selectors": [...],
            "selector_expansion_note": "..."
        }
    """
    return {
        "core": explicit_columns[:max_explicit],
        "selectors": selectors,
        "selector_expansion_note": (
            f"Selectors cover additional columns beyond the {len(explicit_columns[:max_explicit])} "
            f"explicit features. Use expand_feature_selectors() to get full list."
        ) if selectors else None
    }


def validate_selectors_coverage(
    df_columns: List[str],
    explicit_features: List[str],
    selectors: List[Dict[str, Any]],
    expected_coverage: float = 0.95
) -> Dict[str, Any]:
    """
    Validate that selectors + explicit features cover expected columns.

    Args:
        df_columns: All columns
        explicit_features: Explicit feature list
        selectors: Feature selectors
        expected_coverage: Expected fraction of columns covered

    Returns:
        {
            "covered_count": N,
            "total_count": M,
            "coverage": float,
            "meets_threshold": bool,
            "uncovered": [...]
        }
    """
    all_features = set(explicit_features)
    all_features.update(expand_feature_selectors(df_columns, selectors))

    covered = all_features & set(df_columns)
    uncovered = set(df_columns) - covered

    coverage = len(covered) / len(df_columns) if df_columns else 1.0

    return {
        "covered_count": len(covered),
        "total_count": len(df_columns),
        "coverage": coverage,
        "meets_threshold": coverage >= expected_coverage,
        "uncovered": sorted(uncovered)[:50]  # Limit output
    }
