from __future__ import annotations

from typing import Any, Dict, List, Optional


_BOOLEAN_LIKE_TRUE = {"true", "t", "yes", "y", "si", "sí", "1"}
_BOOLEAN_LIKE_FALSE = {"false", "f", "no", "n", "0"}
_NUMERIC_DTYPES = {"float", "float64", "float32", "int", "int64", "int32", "integer", "number", "numeric"}
_TEXT_DTYPES = {"object", "string", "str", "category", "categorical", "bool", "boolean"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _dtype_token(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    for token in _NUMERIC_DTYPES:
        if token in text:
            return "numeric"
    for token in _TEXT_DTYPES:
        if token in text:
            return token
    if "date" in text or "time" in text:
        return "datetime"
    return text


def _target_dtype_for(column: str, column_dtype_targets: Dict[str, Any] | None) -> str:
    if not isinstance(column_dtype_targets, dict):
        return ""
    spec = column_dtype_targets.get(column)
    if not isinstance(spec, dict):
        spec = column_dtype_targets.get(column.lower())
    if isinstance(spec, dict):
        return _dtype_token(spec.get("target_dtype") or spec.get("dtype") or spec.get("type"))
    return _dtype_token(spec)


def _top_values(fact: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    raw = fact.get("top_values")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") if item.get("value") is not None else "").strip()
            if value:
                values.append(value)
    dominant = str(fact.get("dominant_value") if fact.get("dominant_value") is not None else "").strip()
    if dominant and dominant not in values:
        values.insert(0, dominant)
    return values[:8]


def _is_boolean_like(values: List[str]) -> bool:
    if not values:
        return False
    normed = {_norm(value) for value in values if str(value).strip()}
    if not normed:
        return False
    return normed.issubset(_BOOLEAN_LIKE_TRUE | _BOOLEAN_LIKE_FALSE)


def _has_labelled_numeric_values(values: List[str]) -> bool:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        has_digit = any(ch.isdigit() for ch in text)
        has_alpha = any(ch.isalpha() for ch in text)
        if has_digit and has_alpha:
            return True
    return False


def build_column_semantic_cast_risk_pack(
    data_quality_shape_pack: Dict[str, Any] | None,
    *,
    column_dtype_targets: Dict[str, Any] | None = None,
    max_items: int = 80,
) -> Dict[str, Any]:
    """Build advisory facts about risky dtype/semantic casts.

    The pack does not approve or reject anything. It highlights places where
    observed value semantics may contradict a later dtype target so planner,
    data engineer, and reviewers can reason about preservation vs conversion.
    """

    pack = data_quality_shape_pack if isinstance(data_quality_shape_pack, dict) else {}
    facts = pack.get("column_facts") if isinstance(pack.get("column_facts"), list) else []
    risks: List[Dict[str, Any]] = []

    for fact in facts:
        if not isinstance(fact, dict):
            continue
        column = str(fact.get("name") or "").strip()
        if not column:
            continue
        values = _top_values(fact)
        type_hint = _norm(fact.get("type_hint"))
        target_dtype = _target_dtype_for(column, column_dtype_targets)
        text = fact.get("text") if isinstance(fact.get("text"), dict) else {}
        numeric_like_ratio = text.get("numeric_like_ratio")

        reasons: List[str] = []
        suggested_reasoning: List[str] = []

        if _is_boolean_like(values):
            reasons.append("boolean_like_values")
            suggested_reasoning.append("Map boolean labels explicitly (for example TRUE/FALSE -> 1/0) or preserve as categorical; do not use blind numeric coercion.")
        if _has_labelled_numeric_values(values):
            reasons.append("labelled_numeric_values")
            suggested_reasoning.append("If labels contain both codes and text, preserve semantic labels or extract codes with an auditable mapping.")
        if type_hint in {"categorical", "text", "object", "unknown"} and target_dtype == "numeric":
            reasons.append("categorical_to_numeric_target")
            suggested_reasoning.append("Justify numeric target from observed values before casting; otherwise treat as categorical/encoded feature.")
        if numeric_like_ratio is not None:
            try:
                ratio = float(numeric_like_ratio)
            except Exception:
                ratio = None
            if ratio is not None and 0.0 < ratio < 1.0:
                reasons.append("mixed_numeric_text_parseability")
                suggested_reasoning.append("Use non-destructive parsing with coercion-failure accounting; avoid silent null inflation.")

        if not reasons:
            continue

        risks.append(
            {
                "column": column,
                "risk_kinds": list(dict.fromkeys(reasons)),
                "observed_type_hint": fact.get("type_hint"),
                "target_dtype": target_dtype or None,
                "top_values": values,
                "top_value_share": fact.get("top_value_share"),
                "missing_frac": fact.get("missing_frac"),
                "numeric_like_ratio": numeric_like_ratio,
                "suggested_reasoning": list(dict.fromkeys(suggested_reasoning)),
                "severity": "advisory",
            }
        )
        if len(risks) >= max(1, int(max_items)):
            break

    return {
        "schema_version": "1.0",
        "role": "advisory_context_only",
        "deterministic_policy": (
            "This pack exposes observed semantic-cast risks for LLM reasoning. "
            "It must not reject, approve, drop, or cast columns by itself."
        ),
        "risk_count": len(risks),
        "risks": risks,
        "agent_usage_guidance": {
            "execution_planner": "Use to avoid declaring dtype targets or cleaning runbook steps that would erase observed business semantics.",
            "data_engineer": "Use to choose explicit mappings and coercion-failure logging instead of blind casts.",
            "reviewers": "Use to ask whether submitted transformations preserved signal and documented any unavoidable loss.",
        },
    }


def summarize_column_semantic_cast_risk_pack(
    pack: Dict[str, Any] | None,
    *,
    max_items: int = 20,
) -> str:
    if not isinstance(pack, dict) or not pack:
        return ""
    risks = pack.get("risks") if isinstance(pack.get("risks"), list) else []
    lines = [
        "COLUMN_SEMANTIC_CAST_RISK_PACK_SUMMARY:",
        "- role: advisory_context_only; semantic cast facts for senior LLM reasoning, not automatic gates",
        f"- risk_count: {pack.get('risk_count', len(risks))}",
    ]
    for item in risks[: max(0, int(max_items))]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "  "
            + str(
                {
                    "column": item.get("column"),
                    "risk_kinds": item.get("risk_kinds"),
                    "observed_type_hint": item.get("observed_type_hint"),
                    "target_dtype": item.get("target_dtype"),
                    "top_values": item.get("top_values"),
                    "top_value_share": item.get("top_value_share"),
                }
            )
        )
    return "\n".join(lines)
