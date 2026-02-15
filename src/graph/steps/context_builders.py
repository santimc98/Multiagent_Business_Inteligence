"""
Context-builder helpers extracted from graph.py.

Public API
----------
- _norm_name
- _build_required_sample_context
- _infer_parsing_hints_from_sample_context
- _build_signal_summary_context
- _build_cleaned_data_summary_min

Internal helpers (used by the above)
-------------------------------------
- _load_json_safe
- _extract_manifest_row_count
- _estimate_row_count
- _compute_exact_non_null_ratio
- _invert_column_roles
- _pick_split_column
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd

from src.utils.cleaning_validation import sample_raw_columns
from src.utils.contract_accessors import get_canonical_columns, get_column_roles


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())


def _load_json_safe(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Row-count helpers (used by _build_signal_summary_context)
# ---------------------------------------------------------------------------

def _extract_manifest_row_count(manifest: Dict[str, Any]) -> int | None:
    if not isinstance(manifest, dict):
        return None
    row_counts = manifest.get("row_counts")
    if isinstance(row_counts, dict):
        for key in ("after_cleaning", "final", "output", "cleaned", "rows"):
            value = row_counts.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
    for key in ("row_count", "rows", "n_rows", "cleaned_row_count", "output_row_count"):
        value = manifest.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def _estimate_row_count(
    csv_path: str,
    encoding: str = "utf-8",
    sep: str = ",",
    quotechar: str = '"',
    escapechar: str | None = None,
) -> int | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    if not isinstance(sep, str) or len(sep) != 1:
        sep = ","
    if not isinstance(quotechar, str) or len(quotechar) != 1:
        quotechar = '"'
    if not isinstance(escapechar, str) or len(escapechar) != 1:
        escapechar = None

    encodings = [encoding, "utf-8-sig", "latin-1"]
    tried: set[str] = set()
    for enc in encodings:
        if not enc or enc in tried:
            continue
        tried.add(enc)
        try:
            with open(csv_path, "r", encoding=enc, errors="replace", newline="") as f:
                reader = csv.reader(
                    f,
                    delimiter=sep,
                    quotechar=quotechar,
                    escapechar=escapechar,
                )
                next(reader, None)  # header
                return sum(1 for _ in reader)
        except Exception:
            continue
    return None


def _compute_exact_non_null_ratio(
    csv_path: str,
    dialect: Dict[str, Any],
    column_name: str,
    chunk_size: int = 50000,
) -> float | None:
    if not csv_path or not column_name or not os.path.exists(csv_path):
        return None
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    total = 0
    non_null = 0
    null_tokens = {"", "na", "n/a", "nan", "null", "none", "nat"}
    try:
        for chunk in pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            usecols=[column_name],
            chunksize=chunk_size,
            dtype=str,
            low_memory=False,
        ):
            if column_name not in chunk.columns:
                continue
            series = chunk[column_name]
            mask = series.isna()
            try:
                lowered = series.astype(str).str.strip().str.lower()
                mask = mask | lowered.isin(null_tokens)
            except Exception:
                pass
            total += int(series.shape[0])
            non_null += int((~mask).sum())
    except Exception:
        return None
    if total <= 0:
        return None
    return float(non_null / total)


# ---------------------------------------------------------------------------
# Column-role helpers (used by _build_cleaned_data_summary_min)
# ---------------------------------------------------------------------------

def _invert_column_roles(column_roles: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build column -> roles mapping from contract column_roles payload."""
    inverted: Dict[str, List[str]] = {}
    if not isinstance(column_roles, dict):
        return inverted
    for role, cols in column_roles.items():
        if not isinstance(cols, list):
            continue
        for col in cols:
            if not col:
                continue
            key = str(col)
            roles = inverted.setdefault(key, [])
            if role not in roles:
                roles.append(str(role))
    return inverted


def _pick_split_column(df: pd.DataFrame, column_roles: Dict[str, Any]) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    candidates: List[str] = []
    if isinstance(column_roles, dict):
        split_cols = column_roles.get("split_indicator")
        if isinstance(split_cols, list):
            candidates.extend([str(c) for c in split_cols if c])
    if "__split" not in candidates:
        candidates.append("__split")
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


# ---------------------------------------------------------------------------
# Main context-builder functions
# ---------------------------------------------------------------------------

def _build_required_sample_context(
    csv_path: str,
    dialect: Dict[str, Any],
    required_cols: List[str],
    norm_map: Dict[str, str],
    max_rows: int = 50,
    max_examples: int = 6,
) -> str:
    if not csv_path or not required_cols:
        return ""
    raw_cols = []
    canon_to_raw: Dict[str, str] = {}
    for col in required_cols:
        normed = _norm_name(col)
        raw = norm_map.get(normed)
        if raw:
            canon_to_raw[col] = raw
            raw_cols.append(raw)
    if not raw_cols:
        return ""
    sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=max_rows, dtype=str)
    if sample_df is None or getattr(sample_df, "empty", False):
        return ""

    def _pattern_stats(series) -> Dict[str, float]:
        try:
            series = series.dropna().astype(str)
        except Exception:
            return {}
        if series.empty:
            return {}
        sample = series
        percent_like = float(sample.str.contains("%").mean())
        comma_decimal = float(sample.str.contains(r"\d+,\d+").mean())
        dot_decimal = float(sample.str.contains(r"\d+\.\d+").mean())
        numeric_like = float(sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean())
        whitespace = float(sample.str.contains(r"^\s+|\s+$").mean())
        return {
            "numeric_like_ratio": round(numeric_like, 4),
            "percent_like_ratio": round(percent_like, 4),
            "comma_decimal_ratio": round(comma_decimal, 4),
            "dot_decimal_ratio": round(dot_decimal, 4),
            "whitespace_ratio": round(whitespace, 4),
        }

    samples: Dict[str, Dict[str, Any]] = {}
    for canon, raw in canon_to_raw.items():
        if raw not in sample_df.columns:
            continue
        series = sample_df[raw]
        try:
            if series.dtype == object:
                series = series.astype(str).str.strip()
            values = [v for v in series.dropna().tolist() if str(v).strip() != ""]
        except Exception:
            values = []
        uniq: List[str] = []
        for val in values:
            sval = str(val)
            if sval not in uniq:
                uniq.append(sval)
            if len(uniq) >= max_examples:
                break
        samples[canon] = {
            "raw_column": raw,
            "examples": uniq,
            "pattern_stats": _pattern_stats(series),
        }

    if not samples:
        return ""
    payload = {"sample_rows": int(len(sample_df)), "columns": samples}
    return "RAW_REQUIRED_COLUMN_SAMPLES:\n" + json.dumps(payload, ensure_ascii=True)


def _infer_parsing_hints_from_sample_context(sample_context: str) -> str:
    """
    Builds compact, universal parsing guidance from RAW_REQUIRED_COLUMN_SAMPLES.
    This is intentionally generic: it does not hardcode dataset-specific fixes, it derives hints from patterns.
    """
    if not sample_context:
        return ""
    if "RAW_REQUIRED_COLUMN_SAMPLES:" not in sample_context:
        return ""
    try:
        _, raw_json = sample_context.split("\n", 1)
    except ValueError:
        return ""
    try:
        payload = json.loads(raw_json.strip())
    except Exception:
        return ""
    columns = payload.get("columns") if isinstance(payload, dict) else None
    if not isinstance(columns, dict) or not columns:
        return ""

    hints: List[str] = []
    saw_symbols = False
    saw_multi_dot = False
    saw_multi_comma = False
    saw_percent = False

    for meta in columns.values():
        if not isinstance(meta, dict):
            continue
        examples = meta.get("examples") or []
        for ex in examples:
            s = str(ex)
            if "%" in s:
                saw_percent = True
            if s.count(".") >= 2:
                saw_multi_dot = True
            if s.count(",") >= 2:
                saw_multi_comma = True
            # Anything other than digits, whitespace, separators, sign, parentheses, and percent.
            if re.search(r"[^\d\s,.\-+()%]", s):
                saw_symbols = True

    if saw_symbols:
        hints.append("Strip currency symbols/letters before numeric conversion (keep digits, sign, separators, parentheses, and %).")
    if saw_multi_dot:
        hints.append("Values with multiple '.' are usually thousands separators; remove all '.' (unless you also detect a decimal separator elsewhere).")
    if saw_multi_comma:
        hints.append("Values with multiple ',' are usually thousands separators; remove all ',' and infer decimal by the last separator.")
    if saw_percent:
        hints.append("For percentages: strip '%' and normalize 1\u2013100 to 0\u20131.")
    hints.append("If raw values are mostly non-null but conversion yields mostly NaN, treat it as a parsing failure and switch to a more permissive sanitizer.")
    hints.append("Add a small parser self-check: parse the provided examples and confirm you do not get all-NaN for required numeric columns.")

    return "DE_PARSING_HINTS:\n- " + "\n- ".join(hints)


def _build_signal_summary_context(
    csv_path: str,
    dialect: Dict[str, Any],
    required_cols: List[str],
    norm_map: Dict[str, str],
    header_cols: List[str],
    dataset_semantics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    manifest = _load_json_safe("data/cleaning_manifest.json")
    row_count = _extract_manifest_row_count(manifest) or _estimate_row_count(
        csv_path,
        dialect.get("encoding", "utf-8"),
        str(dialect.get("sep") or ","),
        str(dialect.get("quotechar") or '"'),
        dialect.get("escapechar"),
    )
    if row_count is not None:
        summary["row_count"] = row_count
    if header_cols:
        summary["column_count"] = len(header_cols)
    target_canon = None
    target_raw = None
    if isinstance(dataset_semantics, dict):
        target_analysis = dataset_semantics.get("target_analysis") or {}
        if target_analysis.get("partial_label_detected"):
            target_canon = target_analysis.get("primary_target")
            if target_canon:
                target_norm = _norm_name(str(target_canon))
                target_raw = norm_map.get(target_norm, target_canon if target_canon in header_cols else None)
    if required_cols:
        summary["required_columns"] = required_cols
        summary["required_feature_count"] = len(required_cols)
        if row_count:
            summary["rows_per_required_feature"] = round(row_count / max(len(required_cols), 1), 2)
        raw_cols = []
        canon_to_raw: Dict[str, str] = {}
        for col in required_cols:
            normed = _norm_name(col)
            raw = norm_map.get(normed)
            if raw:
                canon_to_raw[col] = raw
                raw_cols.append(raw)
        sample_df = sample_raw_columns(csv_path, dialect, raw_cols, nrows=200, dtype=str)
        if sample_df is not None and not sample_df.empty:
            stats: Dict[str, Any] = {}
            for canon, raw in canon_to_raw.items():
                if raw not in sample_df.columns:
                    continue
                series = sample_df[raw]
                try:
                    values = series.astype(str)
                    non_null_ratio = float(values.replace("", None).notna().mean())
                    nunique = int(values.dropna().nunique())
                    examples = [str(v) for v in values.dropna().head(3).tolist()]
                except Exception:
                    non_null_ratio = None
                    nunique = None
                    examples = []
                stats[canon] = {
                    "raw_column": raw,
                    "sample_non_null_ratio": non_null_ratio,
                    "sample_nunique": nunique,
                    "sample_examples": examples,
                }
            if stats:
                if target_raw and target_canon:
                    exact_ratio = None
                    exact_total = None
                    exact_missing = None
                    if isinstance(dataset_semantics, dict):
                        target_info = dataset_semantics.get("target_analysis") or {}
                        if isinstance(target_info, dict) and target_info.get("target_null_frac_exact") is not None:
                            exact_ratio = 1.0 - float(target_info.get("target_null_frac_exact"))
                            exact_total = target_info.get("target_total_count_exact")
                            exact_missing = target_info.get("target_missing_count_exact")
                    if exact_ratio is None:
                        exact_ratio = _compute_exact_non_null_ratio(csv_path, dialect, target_raw)
                    if exact_ratio is not None:
                        for canon, payload in stats.items():
                            if payload.get("raw_column") == target_raw or canon == target_canon:
                                payload.pop("sample_non_null_ratio", None)
                                payload["exact_non_null_ratio"] = exact_ratio
                                if exact_total is not None:
                                    payload["exact_total_count"] = exact_total
                                if exact_missing is not None:
                                    payload["exact_missing_count"] = exact_missing
                                break
                        else:
                            summary["target_column_stats"] = {
                                "column": target_canon,
                                "raw_column": target_raw,
                                "exact_non_null_ratio": exact_ratio,
                            }
                            if exact_total is not None:
                                summary["target_column_stats"]["exact_total_count"] = exact_total
                            if exact_missing is not None:
                                summary["target_column_stats"]["exact_missing_count"] = exact_missing
                summary["required_column_sample_stats"] = stats
        required_raw = set(canon_to_raw.values())
        candidate_cols = [col for col in header_cols if col not in required_raw]
        if candidate_cols:
            summary["candidate_column_count"] = len(candidate_cols)
            sample_candidates = candidate_cols[:40]
            candidate_df = sample_raw_columns(csv_path, dialect, sample_candidates, nrows=200, dtype=str)
            if candidate_df is not None and not candidate_df.empty:
                candidate_stats: List[Dict[str, Any]] = []
                for col in sample_candidates:
                    if col not in candidate_df.columns:
                        continue
                    series = candidate_df[col]
                    try:
                        values = series.astype(str)
                        non_null_ratio = float(values.replace("", None).notna().mean())
                        nunique = int(values.dropna().nunique())
                        examples = [str(v) for v in values.dropna().head(3).tolist()]
                    except Exception:
                        non_null_ratio = None
                        nunique = None
                        examples = []
                    likely_id = False
                    try:
                        sample_len = len(values.dropna())
                        if sample_len > 0 and nunique is not None:
                            likely_id = nunique >= max(int(sample_len * 0.9), 25)
                    except Exception:
                        likely_id = False
                    candidate_stats.append(
                        {
                            "column": col,
                            "sample_non_null_ratio": non_null_ratio,
                            "sample_nunique": nunique,
                            "sample_examples": examples,
                            "likely_id": likely_id,
                        }
                    )
                if candidate_stats:
                    summary["candidate_columns"] = candidate_stats
    if target_raw and target_canon:
        has_exact = False
        stats_payload = summary.get("required_column_sample_stats")
        if isinstance(stats_payload, dict):
            for payload in stats_payload.values():
                if isinstance(payload, dict) and "exact_non_null_ratio" in payload:
                    has_exact = True
                    break
        if not has_exact and "target_column_stats" not in summary:
            exact_ratio = _compute_exact_non_null_ratio(csv_path, dialect, target_raw)
            if exact_ratio is not None:
                summary["target_column_stats"] = {
                    "column": target_canon,
                    "raw_column": target_raw,
                    "exact_non_null_ratio": exact_ratio,
                }
    return summary


def _build_cleaned_data_summary_min(
    df_clean: pd.DataFrame,
    contract: Dict[str, Any] | None,
    required_columns: List[str] | None,
    data_path: str | None = None,
    max_columns: int = 120,
    outlier_policy: Dict[str, Any] | None = None,
    outlier_report: Dict[str, Any] | None = None,
    cleaning_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build compact cleaned-data facts for ML context without duplicating full profiling.
    The summary is deterministic, small, and derived from DE output + contract decisions.
    """
    contract = contract if isinstance(contract, dict) else {}
    required_columns = [str(col) for col in (required_columns or []) if col]
    column_roles = get_column_roles(contract)
    canonical_columns = get_canonical_columns(contract)
    role_by_column = _invert_column_roles(column_roles if isinstance(column_roles, dict) else {})

    df_cols = [str(c) for c in list(df_clean.columns)]
    df_col_set = set(df_cols)

    # Prioritize contract columns for compact context. Fall back to full columns for narrow datasets.
    priority: List[str] = []
    for seq in (
        required_columns,
        canonical_columns if isinstance(canonical_columns, list) else [],
        list(role_by_column.keys()),
    ):
        for col in seq:
            col_s = str(col)
            if col_s and col_s not in priority:
                priority.append(col_s)
    if len(df_cols) <= max_columns:
        columns_for_summary = list(df_cols)
    else:
        columns_for_summary = [col for col in priority if col in df_col_set]
        for col in df_cols:
            if col in columns_for_summary:
                continue
            columns_for_summary.append(col)
            if len(columns_for_summary) >= max_columns:
                break

    split_column = _pick_split_column(df_clean, column_roles if isinstance(column_roles, dict) else {})
    train_mask = None
    test_mask = None
    train_rows = None
    test_rows = None
    if split_column and split_column in df_clean.columns:
        try:
            split_series = df_clean[split_column].astype(str).str.strip().str.lower()
            train_mask = split_series.eq("train")
            test_mask = split_series.eq("test")
            train_rows = int(train_mask.sum())
            test_rows = int(test_mask.sum())
        except Exception:
            train_mask = None
            test_mask = None
            train_rows = None
            test_rows = None

    objective_type = (
        str(contract.get("objective_type") or "")
        or str((contract.get("objective_analysis") or {}).get("problem_type") or "")
        or str((contract.get("evaluation_spec") or {}).get("objective_type") or "")
    ).strip().lower()

    def _role_dtype_warning(roles: List[str], series: pd.Series) -> str | None:
        if not roles:
            return None
        roles_set = {str(r) for r in roles}
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_object = pd.api.types.is_object_dtype(series)
        if "temporal_features" in roles_set and not (is_datetime or is_object):
            return "temporal_role_with_non_temporal_dtype"
        if ("numerical_features" in roles_set or "spatial_features" in roles_set) and not is_numeric:
            return "numeric_role_with_non_numeric_dtype"
        if any(r in roles_set for r in ("categorical_features", "id", "split_indicator")) and is_numeric:
            return "categorical_role_with_numeric_dtype"
        if "target" in roles_set and objective_type in {"regression", "forecasting", "time_series"} and not is_numeric:
            return "target_role_with_non_numeric_dtype_for_regression"
        return None

    def _masked_null_frac(series: pd.Series, mask: pd.Series | None) -> float | None:
        if mask is None:
            return None
        try:
            sliced = series[mask]
        except Exception:
            return None
        if sliced.shape[0] == 0:
            return None
        try:
            return round(float(sliced.isna().mean()), 6)
        except Exception:
            return None

    rows_total = int(len(df_clean))
    missing_required = [col for col in required_columns if col not in df_col_set]
    column_summaries: List[Dict[str, Any]] = []
    role_dtype_warnings: List[Dict[str, Any]] = []

    for col in columns_for_summary:
        present = col in df_col_set
        expected_roles = role_by_column.get(col, [])
        summary: Dict[str, Any] = {
            "column_name": col,
            "present": bool(present),
            "dtype_observed": None,
            "null_frac": None,
            "in_train": None,
            "in_test": None,
            "mismatch_with_contract": {
                "expected_in_required_columns": col in required_columns,
                "required_but_missing": bool(col in required_columns and not present),
                "expected_roles": expected_roles,
                "role_dtype_warning": None,
            },
        }
        if present:
            series = df_clean[col]
            try:
                summary["dtype_observed"] = str(series.dtype)
            except Exception:
                summary["dtype_observed"] = None
            try:
                summary["null_frac"] = round(float(series.isna().mean()), 6)
            except Exception:
                summary["null_frac"] = None
            if train_mask is not None:
                summary["in_train"] = {
                    "rows": train_rows,
                    "null_frac": _masked_null_frac(series, train_mask),
                }
            if test_mask is not None:
                summary["in_test"] = {
                    "rows": test_rows,
                    "null_frac": _masked_null_frac(series, test_mask),
                }
            warning = _role_dtype_warning(expected_roles, series)
            if warning:
                summary["mismatch_with_contract"]["role_dtype_warning"] = warning
                role_dtype_warnings.append(
                    {"column": col, "warning": warning, "expected_roles": expected_roles, "dtype": summary["dtype_observed"]}
                )
        column_summaries.append(summary)

    omitted_columns_count = max(0, len(df_cols) - len(columns_for_summary))
    outlier_summary: Dict[str, Any] = {}
    if isinstance(outlier_policy, dict) and outlier_policy:
        enabled_raw = outlier_policy.get("enabled")
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}
        elif enabled_raw is None:
            enabled = bool(
                outlier_policy.get("target_columns")
                or outlier_policy.get("methods")
                or outlier_policy.get("treatment")
            )
        else:
            enabled = bool(enabled_raw)
        outlier_summary = {
            "enabled": bool(enabled),
            "apply_stage": str(outlier_policy.get("apply_stage") or "data_engineer").strip().lower(),
            "target_columns": [str(col) for col in (outlier_policy.get("target_columns") or []) if col],
            "report_present": bool(isinstance(outlier_report, dict) and outlier_report),
        }
        if isinstance(outlier_report, dict) and outlier_report:
            for key in ("status", "columns_touched", "rows_affected", "flags_created"):
                if key in outlier_report:
                    outlier_summary[key] = outlier_report.get(key)
        manifest_outlier = (
            cleaning_manifest.get("outlier_treatment")
            if isinstance(cleaning_manifest, dict) and isinstance(cleaning_manifest.get("outlier_treatment"), dict)
            else None
        )
        if manifest_outlier:
            outlier_summary["manifest_outlier_treatment"] = manifest_outlier

    return {
        "version": "v1",
        "source": "system_after_data_engineer",
        "advisory_only": True,
        "contract_precedence_policy": "if_conflict_use_ml_view_and_execution_contract",
        "data_path": str(data_path or "data/cleaned_data.csv"),
        "row_count": rows_total,
        "column_count": int(len(df_cols)),
        "split_column": split_column,
        "required_columns": required_columns,
        "missing_required_columns": missing_required,
        "omitted_columns_count": omitted_columns_count,
        "role_dtype_warnings": role_dtype_warnings,
        "column_summaries": column_summaries,
        "outlier_treatment": outlier_summary,
    }
