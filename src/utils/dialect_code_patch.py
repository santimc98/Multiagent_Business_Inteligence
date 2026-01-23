"""
AST-based autopatcher for pd.read_csv dialect parameters.

This module provides deterministic, minimal patching of Python code to ensure
pd.read_csv calls include the correct sep/decimal/encoding parameters.

NOT a heuristic or dataset-specific fix - purely mechanical AST transformation.
"""

import ast
from typing import List, Tuple


def patch_read_csv_dialect(
    code: str,
    csv_sep: str,
    csv_decimal: str,
    csv_encoding: str,
    expected_path: str = "data/raw.csv",
) -> Tuple[str, List[str], bool]:
    """
    Autopatch pd.read_csv to include correct dialect parameters.

    This function:
    1. Parses code as AST
    2. Finds the target read_csv call (one reading expected_path, or first one)
    3. For that call:
       - If sep/decimal/encoding is missing (and no **kwargs), adds it
       - If sep/decimal/encoding is a literal with wrong value, replaces it
       - Does NOT touch non-literal values (variables/expressions)
       - Does NOT try to modify **kwargs content

    Args:
        code: Python source code to patch
        csv_sep: Expected separator (e.g., ",", ";", "\\t")
        csv_decimal: Expected decimal (e.g., ".", ",")
        csv_encoding: Expected encoding (e.g., "utf-8")
        expected_path: Path to match for target read_csv selection

    Returns:
        Tuple of (patched_code, patch_notes, changed_flag):
        - patched_code: The patched source (or original if no changes/error)
        - patch_notes: List of notes describing what was patched
        - changed_flag: True if any changes were made
    """
    patch_notes: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code, [f"AST parse error: {e}"], False

    # Find all pd.read_csv calls
    calls: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id == "pd":
                    calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "read_csv":
                calls.append(node)

    if not calls:
        return code, ["No pd.read_csv calls found"], False

    # Select target call: prefer one reading expected_path, else first
    def _is_expected_path(arg: ast.AST) -> bool:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value == expected_path
        return False

    target_call = None
    for call in calls:
        if call.args and _is_expected_path(call.args[0]):
            target_call = call
            break
    if target_call is None:
        target_call = calls[0]

    # Check for **kwargs (keyword with arg=None)
    has_kwargs = any(kw.arg is None for kw in target_call.keywords)

    # Build keyword map
    kw_map = {kw.arg: kw for kw in target_call.keywords if kw.arg is not None}

    # Expected parameters
    expected = {
        "sep": csv_sep,
        "decimal": csv_decimal,
        "encoding": csv_encoding,
    }

    changed = False

    def _normalize_encoding(value: str) -> str:
        return str(value).strip().lower().replace("_", "-")

    def _encoding_matches(expected_value: str, actual_value: str) -> bool:
        exp = _normalize_encoding(expected_value)
        act = _normalize_encoding(actual_value)
        if exp in {"utf-8", "utf8"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        if exp in {"utf-8-sig", "utf8-sig"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        return exp == act

    for param, expected_value in expected.items():
        if param not in kw_map:
            # Parameter missing
            if has_kwargs:
                # **kwargs might supply it, don't add
                continue
            # Add the parameter
            new_kw = ast.keyword(arg=param, value=ast.Constant(value=expected_value))
            target_call.keywords.append(new_kw)
            patch_notes.append(f"Added {param}={repr(expected_value)}")
            changed = True
        else:
            # Parameter exists, check if it's a literal mismatch
            kw = kw_map[param]
            val_node = kw.value
            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                actual_value = val_node.value
                if param == "encoding":
                    if not _encoding_matches(expected_value, actual_value):
                        kw.value = ast.Constant(value=expected_value)
                        patch_notes.append(
                            f"Replaced {param}={repr(actual_value)} with {repr(expected_value)}"
                        )
                        changed = True
                else:
                    if actual_value != expected_value:
                        kw.value = ast.Constant(value=expected_value)
                        patch_notes.append(
                            f"Replaced {param}={repr(actual_value)} with {repr(expected_value)}"
                        )
                        changed = True
            # Non-literal values (variables/expressions) are left untouched

    if not changed:
        return code, patch_notes or ["No changes needed"], False

    # Re-serialize the AST
    try:
        patched_code = ast.unparse(tree)
    except Exception as e:
        return code, [f"AST unparse error: {e}"], False

    return patched_code, patch_notes, True


def has_kwargs_in_read_csv(code: str, expected_path: str = "data/raw.csv") -> bool:
    """
    Check if the target read_csv call has **kwargs.

    Args:
        code: Python source code
        expected_path: Path to match for target read_csv selection

    Returns:
        True if target read_csv has **kwargs (keyword with arg=None)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    calls: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id == "pd":
                    calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "read_csv":
                calls.append(node)

    if not calls:
        return False

    def _is_expected_path(arg: ast.AST) -> bool:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value == expected_path
        return False

    target_call = None
    for call in calls:
        if call.args and _is_expected_path(call.args[0]):
            target_call = call
            break
    if target_call is None:
        target_call = calls[0]

    return any(kw.arg is None for kw in target_call.keywords)
