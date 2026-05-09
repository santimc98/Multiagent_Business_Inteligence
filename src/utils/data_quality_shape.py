from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _ratio(count: Optional[int], total: Optional[int]) -> Optional[float]:
    if count is None or total is None or total <= 0:
        return None
    try:
        return round(float(count) / float(total), 6)
    except Exception:
        return None


def _sorted_top(items: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    return sorted(
        [item for item in items if _safe_float(item.get(key)) is not None],
        key=lambda item: float(item.get(key) or 0.0),
        reverse=True,
    )[: max(0, int(limit))]


def _top_value_stats(card_entry: Any, rows: Optional[int]) -> Tuple[Optional[str], Optional[int], Optional[float], List[Dict[str, Any]]]:
    if not isinstance(card_entry, dict):
        return None, None, None, []
    raw_top = card_entry.get("top_values")
    top_values: List[Dict[str, Any]] = []
    if isinstance(raw_top, list):
        for item in raw_top[:5]:
            if not isinstance(item, dict):
                continue
            count = _safe_int(item.get("count"))
            share = _ratio(count, rows)
            top_values.append(
                {
                    "value": str(item.get("value") if item.get("value") is not None else ""),
                    "count": count,
                    "share": share,
                }
            )
    if not top_values:
        return None, None, None, []
    first = top_values[0]
    return str(first.get("value") or ""), _safe_int(first.get("count")), _safe_float(first.get("share")), top_values


def _quality_questions_for_column(fact: Dict[str, Any]) -> List[str]:
    questions: List[str] = []
    name = str(fact.get("name") or "")
    zero_frac = _safe_float((fact.get("numeric") or {}).get("zero_frac") if isinstance(fact.get("numeric"), dict) else None)
    missing_frac = _safe_float(fact.get("missing_frac"))
    top_share = _safe_float(fact.get("top_value_share"))
    if zero_frac is not None and missing_frac is not None and zero_frac > 0 and missing_frac > 0:
        questions.append(
            f"For column '{name}', decide whether zero and null represent different business states before imputing."
        )
    if top_share is not None and top_share >= 0.80:
        questions.append(
            f"For column '{name}', assess whether the dominant value is a valid business state, a default fill, or a low-information feature."
        )
    if bool(fact.get("constant_like")):
        questions.append(
            f"For column '{name}', avoid treating a constant or quasi-constant signal as meaningful without a business rationale."
        )
    return questions[:3]


def build_data_quality_shape_pack(
    dataset_profile: Dict[str, Any] | None,
    *,
    data_profile: Dict[str, Any] | None = None,
    data_atlas: Dict[str, Any] | None = None,
    max_columns: int = 300,
) -> Dict[str, Any]:
    """Build an evidence-only data quality pack for senior agent reasoning.

    The pack deliberately avoids pass/fail decisions. It measures distribution
    shape facts that reviewers and engineers can use to reason about nulls,
    zeros, concentration, dispersion, and low-variability variables.
    """

    profile = dataset_profile if isinstance(dataset_profile, dict) else {}
    data_profile = data_profile if isinstance(data_profile, dict) else {}
    data_atlas = data_atlas if isinstance(data_atlas, dict) else {}

    columns = [str(c) for c in (profile.get("columns") or []) if str(c or "").strip()]
    if not columns and isinstance(data_atlas.get("columns"), list):
        columns = [str((c or {}).get("name")) for c in data_atlas.get("columns", []) if isinstance(c, dict) and (c or {}).get("name")]

    rows = _safe_int(profile.get("rows"))
    sampling = profile.get("sampling") if isinstance(profile.get("sampling"), dict) else {}
    total_rows = _safe_int(sampling.get("total_rows_in_file")) or rows
    type_hints = profile.get("type_hints") if isinstance(profile.get("type_hints"), dict) else {}
    missing_frac = profile.get("missing_frac") if isinstance(profile.get("missing_frac"), dict) else {}
    cardinality = profile.get("cardinality") if isinstance(profile.get("cardinality"), dict) else {}
    numeric_summary = profile.get("numeric_summary") if isinstance(profile.get("numeric_summary"), dict) else {}
    text_summary = profile.get("text_summary") if isinstance(profile.get("text_summary"), dict) else {}

    facts: List[Dict[str, Any]] = []
    warnings_by_kind: Dict[str, List[Dict[str, Any]]] = {
        "zero_missing_coexistence": [],
        "high_zero_fraction": [],
        "high_top_value_concentration": [],
        "constant_or_quasi_constant": [],
        "high_missingness": [],
        "numeric_like_text": [],
        "low_numeric_dispersion": [],
    }
    senior_questions: List[str] = []

    for col in columns[: max(1, int(max_columns))]:
        card_entry = cardinality.get(col)
        unique_count = _safe_int(card_entry.get("unique")) if isinstance(card_entry, dict) else None
        dominant_value, dominant_count, top_value_share, top_values = _top_value_stats(card_entry, rows)
        miss = _safe_float(missing_frac.get(col))
        type_hint = str(type_hints.get(col) or "unknown")
        numeric = numeric_summary.get(col) if isinstance(numeric_summary.get(col), dict) else {}
        text = text_summary.get(col) if isinstance(text_summary.get(col), dict) else {}
        zero_frac = _safe_float(numeric.get("zero_frac")) if isinstance(numeric, dict) else None
        q25 = _safe_float(numeric.get("q25")) if isinstance(numeric, dict) else None
        q75 = _safe_float(numeric.get("q75")) if isinstance(numeric, dict) else None
        min_value = _safe_float(numeric.get("min")) if isinstance(numeric, dict) else None
        max_value = _safe_float(numeric.get("max")) if isinstance(numeric, dict) else None

        unique_ratio = _ratio(unique_count, rows)
        constant_like = bool(
            (unique_count is not None and unique_count <= 1)
            or (top_value_share is not None and top_value_share >= 0.995)
        )
        low_dispersion = bool(
            type_hint == "numeric"
            and q25 is not None
            and q75 is not None
            and q25 == q75
            and min_value is not None
            and max_value is not None
            and min_value != max_value
        )

        advisory_warnings: List[str] = []
        if miss is not None and miss >= 0.40:
            advisory_warnings.append("high_missingness")
            warnings_by_kind["high_missingness"].append({"column": col, "missing_frac": miss})
        if zero_frac is not None and zero_frac >= 0.50:
            advisory_warnings.append("high_zero_fraction")
            warnings_by_kind["high_zero_fraction"].append({"column": col, "zero_frac": zero_frac})
        if zero_frac is not None and miss is not None and zero_frac > 0.0 and miss > 0.0:
            advisory_warnings.append("zero_missing_coexistence")
            warnings_by_kind["zero_missing_coexistence"].append(
                {"column": col, "zero_frac": zero_frac, "missing_frac": miss}
            )
        if top_value_share is not None and top_value_share >= 0.80:
            advisory_warnings.append("high_top_value_concentration")
            warnings_by_kind["high_top_value_concentration"].append(
                {
                    "column": col,
                    "top_value": dominant_value,
                    "top_value_share": top_value_share,
                }
            )
        if constant_like:
            advisory_warnings.append("constant_or_quasi_constant")
            warnings_by_kind["constant_or_quasi_constant"].append(
                {
                    "column": col,
                    "unique_count": unique_count,
                    "top_value_share": top_value_share,
                }
            )
        numeric_like_ratio = _safe_float(text.get("numeric_like_ratio")) if isinstance(text, dict) else None
        if numeric_like_ratio is not None and numeric_like_ratio >= 0.80 and type_hint != "numeric":
            advisory_warnings.append("numeric_like_text")
            warnings_by_kind["numeric_like_text"].append(
                {"column": col, "type_hint": type_hint, "numeric_like_ratio": numeric_like_ratio}
            )
        if low_dispersion:
            advisory_warnings.append("low_numeric_dispersion")
            warnings_by_kind["low_numeric_dispersion"].append(
                {"column": col, "q25": q25, "q75": q75, "min": min_value, "max": max_value}
            )

        fact = {
            "name": col,
            "type_hint": type_hint,
            "missing_frac": miss,
            "non_missing_frac": round(1.0 - miss, 6) if miss is not None else None,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "dominant_value": dominant_value,
            "dominant_count": dominant_count,
            "top_value_share": top_value_share,
            "top_values": top_values,
            "constant_like": constant_like,
            "numeric": {
                "count": _safe_int(numeric.get("count")) if isinstance(numeric, dict) else None,
                "mean": _safe_float(numeric.get("mean")) if isinstance(numeric, dict) else None,
                "std": _safe_float(numeric.get("std")) if isinstance(numeric, dict) else None,
                "min": min_value,
                "p01": _safe_float(numeric.get("p01")) if isinstance(numeric, dict) else None,
                "q25": q25,
                "median": _safe_float(numeric.get("median")) if isinstance(numeric, dict) else None,
                "q75": q75,
                "p99": _safe_float(numeric.get("p99")) if isinstance(numeric, dict) else None,
                "max": max_value,
                "zero_frac": zero_frac,
                "neg_frac": _safe_float(numeric.get("neg_frac")) if isinstance(numeric, dict) else None,
                "pos_frac": _safe_float(numeric.get("pos_frac")) if isinstance(numeric, dict) else None,
            } if isinstance(numeric, dict) and numeric else {},
            "text": {
                "empty_frac": _safe_float(text.get("empty_frac")) if isinstance(text, dict) else None,
                "whitespace_frac": _safe_float(text.get("whitespace_frac")) if isinstance(text, dict) else None,
                "numeric_like_ratio": numeric_like_ratio,
                "datetime_like_ratio": _safe_float(text.get("datetime_like_ratio")) if isinstance(text, dict) else None,
            } if isinstance(text, dict) and text else {},
            "advisory_warnings": advisory_warnings,
        }
        senior_questions.extend(_quality_questions_for_column(fact))
        facts.append(fact)

    signal_counts = {kind: len(values) for kind, values in warnings_by_kind.items()}
    shape_signals = {
        "counts": signal_counts,
        "top_missingness": _sorted_top(warnings_by_kind["high_missingness"], "missing_frac", 20),
        "top_zero_fraction": _sorted_top(warnings_by_kind["high_zero_fraction"], "zero_frac", 20),
        "top_value_concentration": _sorted_top(
            warnings_by_kind["high_top_value_concentration"], "top_value_share", 20
        ),
        "zero_missing_coexistence": _sorted_top(
            warnings_by_kind["zero_missing_coexistence"], "zero_frac", 20
        ),
        "constant_or_quasi_constant": warnings_by_kind["constant_or_quasi_constant"][:40],
        "numeric_like_text": _sorted_top(warnings_by_kind["numeric_like_text"], "numeric_like_ratio", 20),
        "low_numeric_dispersion": warnings_by_kind["low_numeric_dispersion"][:40],
    }

    return {
        "schema_version": "1.0",
        "role": "advisory_context_only",
        "deterministic_policy": (
            "This pack measures data-shape evidence for LLM reasoning. It must not reject, approve, "
            "or override an agent decision by itself."
        ),
        "source": {
            "dataset_profile_present": bool(profile),
            "data_profile_present": bool(data_profile),
            "data_atlas_present": bool(data_atlas),
            "rows_profiled": rows,
            "total_rows_in_file": total_rows,
            "columns_profiled": len(columns),
            "sampling": sampling,
        },
        "shape_signals": shape_signals,
        "column_facts": facts,
        "senior_reasoning_questions": list(dict.fromkeys(senior_questions))[:40],
        "agent_usage_guidance": {
            "steward": "Use to characterize zeros, nulls, dispersion, concentration, and low-variability columns before assigning semantic roles.",
            "execution_planner": "Use as factual context when deciding cleaning obligations and evidence requirements; do not convert advisory warnings into hard gates automatically.",
            "data_engineer": "Use to decide imputations, transformations, flags, and cleaning explanations with explicit zero-vs-null reasoning.",
            "ml_engineer": "Use to reason about feature robustness, low-variance signals, imputation strategy, and over-reliance risks.",
            "reviewers": "Use to ask whether the submitted solution addressed material data-shape risks; warnings are evidence, not deterministic failures.",
            "business_translator": "Use to explain data-quality limitations and business interpretation in the executive report.",
        },
    }


def summarize_data_quality_shape_pack(
    pack: Dict[str, Any] | None,
    *,
    max_columns: int = 40,
    max_lines: int = 90,
) -> str:
    if not isinstance(pack, dict) or not pack:
        return ""
    source = pack.get("source") if isinstance(pack.get("source"), dict) else {}
    signals = pack.get("shape_signals") if isinstance(pack.get("shape_signals"), dict) else {}
    counts = signals.get("counts") if isinstance(signals.get("counts"), dict) else {}
    lines: List[str] = [
        "DATA_QUALITY_SHAPE_PACK_SUMMARY:",
        "- role: advisory_context_only; facts for senior LLM reasoning, not deterministic pass/fail gates",
        f"- rows_profiled: {source.get('rows_profiled')}; total_rows_in_file: {source.get('total_rows_in_file')}; columns_profiled: {source.get('columns_profiled')}",
        f"- signal_counts: {counts}",
    ]

    def _append_items(title: str, items: Any, fields: List[str], limit: int = 8) -> None:
        if not isinstance(items, list) or not items:
            return
        preview = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            compact = {field: item.get(field) for field in fields if item.get(field) is not None}
            if compact:
                preview.append(compact)
        if preview:
            lines.append(f"- {title}: {preview}")

    _append_items("top_missingness", signals.get("top_missingness"), ["column", "missing_frac"])
    _append_items("top_zero_fraction", signals.get("top_zero_fraction"), ["column", "zero_frac"])
    _append_items("zero_missing_coexistence", signals.get("zero_missing_coexistence"), ["column", "zero_frac", "missing_frac"])
    _append_items("top_value_concentration", signals.get("top_value_concentration"), ["column", "top_value", "top_value_share"])
    _append_items("constant_or_quasi_constant", signals.get("constant_or_quasi_constant"), ["column", "unique_count", "top_value_share"])
    _append_items("numeric_like_text", signals.get("numeric_like_text"), ["column", "type_hint", "numeric_like_ratio"])
    _append_items("low_numeric_dispersion", signals.get("low_numeric_dispersion"), ["column", "q25", "q75", "min", "max"])

    facts = pack.get("column_facts") if isinstance(pack.get("column_facts"), list) else []
    material_facts = [
        item for item in facts
        if isinstance(item, dict) and item.get("advisory_warnings")
    ][: max(0, int(max_columns))]
    if material_facts:
        lines.append("- material_column_warnings:")
        for item in material_facts[:12]:
            lines.append(
                "  "
                + str(
                    {
                        "column": item.get("name"),
                        "warnings": item.get("advisory_warnings"),
                        "missing_frac": item.get("missing_frac"),
                        "top_value_share": item.get("top_value_share"),
                        "zero_frac": (item.get("numeric") or {}).get("zero_frac")
                        if isinstance(item.get("numeric"), dict)
                        else None,
                    }
                )
            )

    questions = pack.get("senior_reasoning_questions")
    if isinstance(questions, list) and questions:
        lines.append("- senior_reasoning_questions:")
        for q in questions[:10]:
            lines.append(f"  - {str(q)}")

    return "\n".join(lines[: max(1, int(max_lines))])
