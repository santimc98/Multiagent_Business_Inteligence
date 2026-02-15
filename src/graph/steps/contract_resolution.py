"""
Contract resolution utilities extracted from graph.py (seniority refactoring).

Functions for resolving contract columns, required outputs, and expected output paths.
"""

import os
import re
from typing import Any, Dict, List

from src.utils.contract_accessors import (
    get_canonical_columns,
    get_artifact_requirements,
    get_derived_column_names,
    get_required_outputs as accessor_get_required_outputs,
)


def _norm_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())


_REQUIRED_OUTPUT_EXTENSIONS = {
    ".csv", ".json", ".png", ".pdf", ".parquet",
    ".pkl", ".pickle", ".joblib", ".txt", ".md",
}


def _looks_like_filesystem_path(value: str) -> bool:
    if not value:
        return False
    text = str(value).strip()
    if not text:
        return False
    lower = text.lower()
    if lower.startswith(("data/", "static/", "artifacts/")):
        return True
    if "/" in text or "\\" in text:
        return True
    _, ext = os.path.splitext(lower)
    return ext in _REQUIRED_OUTPUT_EXTENSIONS


def _normalize_output_path(path: str) -> str:
    """Normalize separators while preserving contract-declared paths."""
    if not path:
        return path
    return str(path).strip().replace("\\", "/").lstrip("/")


def _is_glob_pattern(path: str) -> bool:
    if not path:
        return False
    return any(ch in path for ch in ["*", "?", "["]) or path.endswith(("/", "\\"))


def _resolve_contract_columns(contract: Dict[str, Any], sources: set[str] | None = None) -> List[str]:
    """V4.1: Use canonical_columns for input, derived_columns for derived. No legacy fallback."""
    if not contract or not isinstance(contract, dict):
        return []
    if sources and 'input' in sources:
        return get_canonical_columns(contract)
    if sources and 'derived' in sources:
        return get_derived_column_names(contract)
    return get_canonical_columns(contract)


def _resolve_allowed_columns_for_gate(
    state: Dict[str, Any],
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any] | None = None,
) -> List[str]:
    allowed: List[str] = []
    csv_path = state.get("ml_data_path")
    if isinstance(csv_path, str) and csv_path and os.path.exists(csv_path):
        try:
            import pandas as pd
            csv_sep = state.get("csv_sep", ",")
            csv_decimal = state.get("csv_decimal", ".")
            csv_encoding = state.get("csv_encoding", "utf-8")
            header_df = pd.read_csv(csv_path, nrows=0, sep=csv_sep, decimal=csv_decimal, encoding=csv_encoding)
            allowed.extend([str(col) for col in header_df.columns.tolist() if col])
        except Exception:
            pass

    if isinstance(contract, dict):
        derived_cols = get_derived_column_names(contract)
        allowed.extend([str(c) for c in derived_cols if c])

    if not allowed:
        profile = state.get("profile") or state.get("dataset_profile")
        if isinstance(profile, dict):
            cols = profile.get("columns")
            if isinstance(cols, list):
                allowed.extend([str(col) for col in cols if col])

    seen: set[str] = set()
    deduped: List[str] = []
    for col in allowed:
        norm = _norm_name(col)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(col)
    return deduped


def _resolve_allowed_patterns_for_gate(contract: Any) -> List[str]:
    """V4.1: Use artifact_requirements.file_schemas only. No legacy fallback."""
    patterns: List[str] = []
    if not isinstance(contract, dict):
        return patterns
    artifact_requirements = contract.get("artifact_requirements", {})
    if not isinstance(artifact_requirements, dict):
        artifact_requirements = {}
    schema = artifact_requirements.get("file_schemas")
    if isinstance(schema, dict):
        scored_schema = schema.get("data/scored_rows.csv")
        if isinstance(scored_schema, dict):
            allowed_patterns = scored_schema.get("allowed_name_patterns")
            if isinstance(allowed_patterns, list):
                patterns.extend([str(pat) for pat in allowed_patterns if isinstance(pat, str) and pat.strip()])
    return patterns


def _resolve_contract_columns_for_cleaning(contract: Dict[str, Any], sources: set[str] | None = None) -> List[str]:
    """V4.1: Use canonical_columns for cleaning. No legacy data_requirements."""
    if not contract or not isinstance(contract, dict):
        return []
    columns = list(get_canonical_columns(contract))
    artifacts = get_artifact_requirements(contract)
    schema_binding = artifacts.get("schema_binding", {})
    if isinstance(schema_binding, dict):
        passthrough = schema_binding.get("optional_passthrough_columns", [])
        if isinstance(passthrough, list):
            columns.extend([str(c) for c in passthrough if c])
    return columns


def _merge_conceptual_outputs(
    state: Dict[str, Any] | None,
    contract: Dict[str, Any],
    conceptual_outputs: List[str],
) -> None:
    if not conceptual_outputs or not isinstance(state, dict):
        return
    reporting = state.get("reporting_requirements")
    if not isinstance(reporting, dict):
        reporting = {}
    merged: List[str] = []
    seen: set[str] = set()

    def _add_items(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)

    contract_reporting = contract.get("reporting_requirements") if isinstance(contract, dict) else None
    _add_items(reporting.get("conceptual_outputs"))
    if isinstance(contract_reporting, dict):
        _add_items(contract_reporting.get("conceptual_outputs"))
    _add_items(conceptual_outputs)

    reporting["conceptual_outputs"] = merged
    state["reporting_requirements"] = reporting


def _resolve_required_outputs(contract: Dict[str, Any], state: Dict[str, Any] | None = None) -> List[str]:
    """
    Resolve required outputs from V4.1 contract structure.
    Priority:
      1. contract.required_outputs (single source of truth)
      2. evaluation_spec.required_outputs (fallback when contract list is absent/empty)
      3. accessor union (legacy fallback only)
    """
    if not isinstance(contract, dict):
        return []

    conceptual_outputs: List[str] = []

    def _split_outputs(values: Any) -> tuple[List[str], List[str]]:
        file_like: List[str] = []
        conceptual_like: List[str] = []
        if isinstance(values, list):
            for item in values:
                if not item:
                    continue
                path = str(item)
                if _looks_like_filesystem_path(path):
                    file_like.append(_normalize_output_path(path))
                else:
                    conceptual_like.append(path)
        return file_like, conceptual_like

    resolved: List[str] = []
    seen: set[str] = set()

    def _add_output(path: str) -> None:
        if not path:
            return
        normalized = _normalize_output_path(str(path))
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        resolved.append(normalized)

    # Priority 1: contract.required_outputs (canonical)
    has_contract_required_outputs = False
    req_outputs = contract.get("required_outputs")
    if isinstance(req_outputs, list) and req_outputs:
        file_like, conceptual_like = _split_outputs(req_outputs)
        conceptual_outputs.extend(conceptual_like)
        for path in file_like:
            _add_output(path)
        has_contract_required_outputs = bool(file_like)

    if has_contract_required_outputs:
        eval_spec = contract.get("evaluation_spec")
        if isinstance(eval_spec, dict):
            eval_outputs = eval_spec.get("required_outputs")
            if isinstance(eval_outputs, list) and eval_outputs:
                _, conceptual_like = _split_outputs(eval_outputs)
                conceptual_outputs.extend(conceptual_like)
        try:
            accessor_outputs = accessor_get_required_outputs(contract)
        except Exception:
            accessor_outputs = []
        if isinstance(accessor_outputs, list):
            for entry in accessor_outputs:
                if not entry:
                    continue
                text = str(entry)
                if not _looks_like_filesystem_path(text):
                    conceptual_outputs.append(text)
        _merge_conceptual_outputs(state, contract, conceptual_outputs)
        return resolved

    # Fallback 1: evaluation_spec.required_outputs
    eval_spec = contract.get("evaluation_spec")
    if isinstance(eval_spec, dict):
        eval_outputs = eval_spec.get("required_outputs")
        if isinstance(eval_outputs, list) and eval_outputs:
            file_like, conceptual_like = _split_outputs(eval_outputs)
            conceptual_outputs.extend(conceptual_like)
            for path in file_like:
                _add_output(path)

    # Fallback 2: accessor union
    try:
        accessor_outputs = accessor_get_required_outputs(contract)
    except Exception:
        accessor_outputs = []
    if isinstance(accessor_outputs, list):
        for entry in accessor_outputs:
            if not entry:
                continue
            text = str(entry)
            if _looks_like_filesystem_path(text):
                _add_output(text)
            else:
                conceptual_outputs.append(text)

    _merge_conceptual_outputs(state, contract, conceptual_outputs)
    return resolved


def _resolve_expected_output_paths(contract: Dict[str, Any], state: Dict[str, Any] | None = None) -> List[str]:
    """V4.1-only: delegates to _resolve_required_outputs."""
    return _resolve_required_outputs(contract, state)


def _resolve_optional_runtime_downloads(contract: Dict[str, Any]) -> List[str]:
    """
    Resolve non-blocking artifacts that should be pulled from sandbox outputs when present.
    """
    if not isinstance(contract, dict):
        return []

    resolved: List[str] = []
    seen: set[str] = set()

    def _add_optional(path: Any) -> None:
        text = str(path or "").strip()
        if not text:
            return
        if not _looks_like_filesystem_path(text):
            return
        normalized = _normalize_output_path(text)
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        resolved.append(normalized)

    artifacts = contract.get("artifact_requirements") if isinstance(contract.get("artifact_requirements"), dict) else {}
    for key in ("optional_outputs", "optional_files"):
        values = artifacts.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    _add_optional(item.get("path") or item.get("output_path") or item.get("file"))
                else:
                    _add_optional(item)

    for path in (
        "data/alignment_check.json",
        "data/metrics.json",
        "data/scored_rows.csv",
        "data/case_alignment_report.json",
    ):
        _add_optional(path)
    return resolved
