from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


_GENERIC_TOKENS = {
    "id",
    "key",
    "code",
    "codigo",
    "cod",
    "num",
    "number",
    "count",
    "value",
    "valor",
    "total",
    "mean",
    "avg",
    "min",
    "max",
    "flag",
    "is",
    "has",
    "the",
    "and",
}


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


def _strip_accents(value: str) -> str:
    try:
        return "".join(
            char for char in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(char)
        )
    except Exception:
        return value


def _split_camel(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", value)


def _name_tokens(name: Any) -> List[str]:
    raw = _strip_accents(_split_camel(str(name or ""))).lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    tokens = [tok for tok in raw.split("_") if tok]
    cleaned: List[str] = []
    for token in tokens:
        token = re.sub(r"\d+$", "", token)
        if len(token) < 3 or token in _GENERIC_TOKENS:
            continue
        cleaned.append(token)
    return cleaned


def _stem_token(token: str) -> str:
    token = token.lower().strip()
    for suffix in ("ciones", "cion", "sion", "mentos", "miento", "acion", "idad", "ados", "adas", "ado", "ada", "es", "os", "as", "s"):
        if len(token) - len(suffix) >= 5 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) >= 8:
        return token[:7]
    if len(token) >= 6:
        return token[:6]
    return token


def _column_stems(name: Any) -> List[str]:
    stems = []
    for token in _name_tokens(name):
        stem = _stem_token(token)
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def _columns_from_profile(dataset_profile: Dict[str, Any], data_profile: Dict[str, Any]) -> List[str]:
    cols = dataset_profile.get("columns")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols if str(c or "").strip()]
    basic = data_profile.get("basic_stats") if isinstance(data_profile.get("basic_stats"), dict) else {}
    cols = basic.get("columns")
    if isinstance(cols, list):
        return [str(c) for c in cols if str(c or "").strip()]
    dtypes = data_profile.get("dtypes")
    if isinstance(dtypes, dict):
        return [str(c) for c in dtypes.keys() if str(c or "").strip()]
    return []


def _semantic_duplicate_groups(columns: List[str], *, max_groups: int = 40) -> List[Dict[str, Any]]:
    stem_to_cols: Dict[str, List[str]] = defaultdict(list)
    for col in columns:
        for stem in _column_stems(col):
            stem_to_cols[stem].append(col)

    groups: List[Dict[str, Any]] = []
    seen_signatures = set()
    for stem, cols in stem_to_cols.items():
        unique_cols = list(dict.fromkeys(cols))
        if len(unique_cols) < 2:
            continue
        signature = tuple(sorted(unique_cols))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        groups.append(
            {
                "semantic_anchor": stem,
                "columns": unique_cols[:20],
                "column_count": len(unique_cols),
                "evidence": "shared_normalized_name_root",
                "interpretation": (
                    "Columns may encode the same business concept at different granularity, source, or encoding. "
                    "A senior agent should decide whether to keep both, select one, or document the rationale."
                ),
            }
        )
    groups.sort(key=lambda item: (int(item.get("column_count") or 0), len(str(item.get("semantic_anchor") or ""))), reverse=True)
    return groups[: max(0, int(max_groups))]


def _family_groups(columns: List[str], *, max_families: int = 40) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for col in columns:
        stems = _column_stems(col)
        if not stems:
            continue
        anchor = stems[0]
        buckets[anchor].append(col)
    families = []
    for anchor, cols in buckets.items():
        unique_cols = list(dict.fromkeys(cols))
        if len(unique_cols) < 2:
            continue
        families.append(
            {
                "family": anchor,
                "columns_sample": unique_cols[:16],
                "column_count": len(unique_cols),
                "derivation": "name_root_family",
            }
        )
    families.sort(key=lambda item: int(item.get("column_count") or 0), reverse=True)
    return families[: max(0, int(max_families))]


def _normalize_corr_pair(item: Dict[str, Any]) -> Dict[str, Any]:
    col_a = item.get("col_a") or item.get("column_a") or item.get("feature_a") or item.get("a")
    col_b = item.get("col_b") or item.get("column_b") or item.get("feature_b") or item.get("b")
    corr = (
        item.get("corr_abs")
        if item.get("corr_abs") is not None
        else item.get("abs_corr")
        if item.get("abs_corr") is not None
        else item.get("correlation")
    )
    return {
        "col_a": str(col_a or ""),
        "col_b": str(col_b or ""),
        "corr_abs": _safe_float(corr),
        "source": str(item.get("source") or "data_profile.multicollinearity_pairs_high"),
    }


def _correlation_pairs(data_profile: Dict[str, Any], *, max_pairs: int = 80) -> List[Dict[str, Any]]:
    raw = data_profile.get("multicollinearity_pairs_high")
    if not isinstance(raw, list):
        raw = []
    pairs = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_corr_pair(item)
        if normalized.get("col_a") and normalized.get("col_b"):
            pairs.append(normalized)
    pairs.sort(key=lambda item: float(item.get("corr_abs") or 0.0), reverse=True)
    return pairs[: max(0, int(max_pairs))]


def _target_association_concentration(data_profile: Dict[str, Any], *, top_n: int = 10) -> Dict[str, Any]:
    raw = data_profile.get("feature_target_associations")
    if not isinstance(raw, list) or not raw:
        return {"available": False, "top_features": []}
    features = []
    total = 0.0
    for item in raw:
        if not isinstance(item, dict):
            continue
        score = _safe_float(item.get("score"))
        if score is None:
            continue
        total += abs(float(score))
        features.append(
            {
                "column": str(item.get("column") or ""),
                "score": score,
                "method": str(item.get("method") or ""),
                "direction": str(item.get("direction") or ""),
            }
        )
    features.sort(key=lambda item: abs(float(item.get("score") or 0.0)), reverse=True)
    if not features:
        return {"available": False, "top_features": []}
    top_score = abs(float(features[0].get("score") or 0.0))
    return {
        "available": True,
        "top_features": features[: max(1, int(top_n))],
        "top_feature_share_of_observed_association": round(top_score / total, 6) if total > 0 else None,
        "top_3_share_of_observed_association": round(
            sum(abs(float(item.get("score") or 0.0)) for item in features[:3]) / total,
            6,
        ) if total > 0 else None,
        "interpretation": (
            "Association scores are univariate evidence, not model importances. Use them to reason about "
            "potential dependence on a small number of concepts before model training."
        ),
    }


def _family_from_column_sets(column_sets: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_sets = column_sets.get("sets") if isinstance(column_sets.get("sets"), list) else []
    out = []
    for item in raw_sets[:40]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("family") or "").strip()
        if not name:
            continue
        out.append(
            {
                "family": name,
                "column_count": _safe_int(item.get("count")),
                "selector": item.get("selector") if isinstance(item.get("selector"), dict) else {},
                "derivation": "column_sets",
            }
        )
    return out


def build_feature_governance_pack(
    dataset_profile: Dict[str, Any] | None,
    *,
    data_profile: Dict[str, Any] | None = None,
    dataset_semantics: Dict[str, Any] | None = None,
    column_sets: Dict[str, Any] | None = None,
    model_card: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    dataset_profile = dataset_profile if isinstance(dataset_profile, dict) else {}
    data_profile = data_profile if isinstance(data_profile, dict) else {}
    dataset_semantics = dataset_semantics if isinstance(dataset_semantics, dict) else {}
    column_sets = column_sets if isinstance(column_sets, dict) else {}
    model_card = model_card if isinstance(model_card, dict) else {}
    columns = _columns_from_profile(dataset_profile, data_profile)

    semantic_duplicate_groups = _semantic_duplicate_groups(columns)
    name_family_groups = _family_groups(columns)
    declared_family_groups = _family_from_column_sets(column_sets)
    correlation_pairs = _correlation_pairs(data_profile)
    association_concentration = _target_association_concentration(data_profile)

    model_importance_context: Dict[str, Any] = {"available": False}
    feature_importances = model_card.get("feature_importance") or model_card.get("feature_importances")
    if isinstance(feature_importances, list) and feature_importances:
        normalized = []
        total = 0.0
        for item in feature_importances:
            if not isinstance(item, dict):
                continue
            value = _safe_float(item.get("importance") or item.get("weight") or item.get("value"))
            if value is None:
                continue
            total += abs(value)
            normalized.append({"feature": str(item.get("feature") or item.get("column") or ""), "importance": value})
        normalized.sort(key=lambda item: abs(float(item.get("importance") or 0.0)), reverse=True)
        if normalized:
            model_importance_context = {
                "available": True,
                "top_features": normalized[:15],
                "top_feature_share": round(abs(float(normalized[0].get("importance") or 0.0)) / total, 6) if total > 0 else None,
                "top_5_share": round(sum(abs(float(item.get("importance") or 0.0)) for item in normalized[:5]) / total, 6) if total > 0 else None,
            }

    senior_questions = []
    if semantic_duplicate_groups:
        senior_questions.append(
            "Do any duplicated semantic concepts double-count the same business signal, and should the model keep one representative or document why both are needed?"
        )
    if correlation_pairs:
        senior_questions.append(
            "Do highly correlated features represent redundant encodings, legitimate complementary views, or leakage-like derived variables?"
        )
    if association_concentration.get("available"):
        senior_questions.append(
            "Is predictive signal concentrated in a small number of variables or families, and what operational dependency risk does that create?"
        )
    if declared_family_groups or name_family_groups:
        senior_questions.append(
            "Should feature selection balance business families rather than optimizing only the strongest individual variables?"
        )

    return {
        "schema_version": "1.0",
        "role": "advisory_context_only",
        "deterministic_policy": (
            "This pack surfaces feature-governance evidence for senior LLM reasoning. It must not reject, approve, "
            "or override an agent decision by itself."
        ),
        "source": {
            "columns_observed": len(columns),
            "dataset_profile_present": bool(dataset_profile),
            "data_profile_present": bool(data_profile),
            "dataset_semantics_present": bool(dataset_semantics),
            "column_sets_present": bool(column_sets),
            "model_card_present": bool(model_card),
        },
        "feature_governance_signals": {
            "semantic_duplicate_groups": semantic_duplicate_groups,
            "high_correlation_pairs": correlation_pairs,
            "declared_feature_families": declared_family_groups,
            "name_inferred_feature_families": name_family_groups,
            "target_association_concentration": association_concentration,
            "model_importance_concentration": model_importance_context,
        },
        "senior_reasoning_questions": senior_questions,
        "agent_usage_guidance": {
            "steward": "Use to identify concept aliases and feature families while assigning semantic roles.",
            "strategist": "Use to propose feature-selection hypotheses that reduce double-counting and operational dependency risk.",
            "execution_planner": "Use as evidence for required reports or reviewer questions; do not turn advisory correlation facts into automatic hard gates.",
            "data_engineer": "Use to preserve concept lineage and avoid accidentally duplicating one concept under multiple names.",
            "ml_engineer": "Use to reason about correlated variables, grouped feature selection, and dominance risk before/after training.",
            "reviewers": "Use to evaluate whether the solution addressed material feature-governance risks; warnings are evidence, not deterministic failures.",
            "business_translator": "Use to explain feature selection, dependency risk, and business limitations in the executive report.",
        },
    }


def summarize_feature_governance_pack(
    pack: Dict[str, Any] | None,
    *,
    max_lines: int = 90,
) -> str:
    if not isinstance(pack, dict) or not pack:
        return ""
    source = pack.get("source") if isinstance(pack.get("source"), dict) else {}
    signals = pack.get("feature_governance_signals") if isinstance(pack.get("feature_governance_signals"), dict) else {}
    duplicate_groups = signals.get("semantic_duplicate_groups") if isinstance(signals.get("semantic_duplicate_groups"), list) else []
    corr_pairs = signals.get("high_correlation_pairs") if isinstance(signals.get("high_correlation_pairs"), list) else []
    declared_families = signals.get("declared_feature_families") if isinstance(signals.get("declared_feature_families"), list) else []
    inferred_families = signals.get("name_inferred_feature_families") if isinstance(signals.get("name_inferred_feature_families"), list) else []
    assoc = signals.get("target_association_concentration") if isinstance(signals.get("target_association_concentration"), dict) else {}
    model_imp = signals.get("model_importance_concentration") if isinstance(signals.get("model_importance_concentration"), dict) else {}

    lines = [
        "FEATURE_GOVERNANCE_PACK_SUMMARY:",
        "- role: advisory_context_only; facts for senior LLM feature-governance reasoning, not deterministic pass/fail gates",
        f"- columns_observed: {source.get('columns_observed')}; data_profile_present: {source.get('data_profile_present')}; model_card_present: {source.get('model_card_present')}",
        f"- counts: semantic_duplicate_groups={len(duplicate_groups)}, high_correlation_pairs={len(corr_pairs)}, declared_families={len(declared_families)}, inferred_families={len(inferred_families)}",
    ]
    if duplicate_groups:
        lines.append("- semantic_duplicate_groups_sample:")
        for item in duplicate_groups[:8]:
            lines.append(
                "  "
                + str(
                    {
                        "anchor": item.get("semantic_anchor"),
                        "columns": item.get("columns"),
                        "column_count": item.get("column_count"),
                    }
                )
            )
    if corr_pairs:
        lines.append("- high_correlation_pairs_sample:")
        for item in corr_pairs[:10]:
            lines.append(
                "  "
                + str(
                    {
                        "col_a": item.get("col_a"),
                        "col_b": item.get("col_b"),
                        "corr_abs": item.get("corr_abs"),
                    }
                )
            )
    if declared_families or inferred_families:
        lines.append(
            f"- feature_families_sample: declared={declared_families[:6]}, inferred={inferred_families[:6]}"
        )
    if assoc.get("available"):
        lines.append(
            f"- target_association_concentration: top_share={assoc.get('top_feature_share_of_observed_association')}, top3_share={assoc.get('top_3_share_of_observed_association')}, top_features={assoc.get('top_features', [])[:6]}"
        )
    if model_imp.get("available"):
        lines.append(
            f"- model_importance_concentration: top_share={model_imp.get('top_feature_share')}, top5_share={model_imp.get('top_5_share')}, top_features={model_imp.get('top_features', [])[:6]}"
        )
    questions = pack.get("senior_reasoning_questions")
    if isinstance(questions, list) and questions:
        lines.append("- senior_reasoning_questions:")
        for question in questions[:8]:
            lines.append(f"  - {question}")
    return "\n".join(lines[: max(1, int(max_lines))])
