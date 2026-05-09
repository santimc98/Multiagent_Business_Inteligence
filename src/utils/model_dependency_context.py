from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


_SOURCE_FAMILY_TOKENS = {
    "identifier_or_key": ("id", "uuid", "key", "entity", "customer", "deudor", "corporation", "account"),
    "public_or_legal": ("legal", "court", "judicial", "public", "registry", "incidence", "incidencia", "forma", "mercantil"),
    "financial_statement": ("ebitda", "liquidity", "solvency", "debt", "sales", "ventas", "balance", "asset", "activo", "financial"),
    "payment_or_behavior": ("payment", "pago", "cobro", "mora", "default", "delay", "retraso", "saldo", "risk", "riesgo"),
    "commercial_classification": ("sector", "segment", "segmentacion", "classification", "coface", "industry", "activity"),
    "temporal": ("date", "fecha", "month", "year", "day", "hour", "snapshot", "period", "time"),
    "geographic": ("geo", "country", "region", "city", "postal", "province", "lat", "lon"),
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", raw)
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    return raw.strip("_")


def _infer_source_family(feature: Any) -> str:
    name = _normalize_name(feature)
    tokens = [tok for tok in name.split("_") if tok]
    for family, patterns in _SOURCE_FAMILY_TOKENS.items():
        if any(pattern in tokens or pattern in name for pattern in patterns):
            return family
    return "unknown_or_domain_specific"


def _coerce_importance_records(payload: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in (
            "feature_importances",
            "feature_importance",
            "features_importance",
            "importance",
            "top_features",
        ):
            nested = payload.get(key)
            if nested is not None:
                records.extend(_coerce_importance_records(nested))
        if payload.get("feature") or payload.get("column") or payload.get("name"):
            value = _safe_float(
                payload.get("importance")
                if payload.get("importance") is not None
                else payload.get("weight")
                if payload.get("weight") is not None
                else payload.get("value")
                if payload.get("value") is not None
                else payload.get("score")
            )
            if value is not None:
                records.append(
                    {
                        "feature": str(payload.get("feature") or payload.get("column") or payload.get("name") or ""),
                        "importance": value,
                        "source_field": "dict_record",
                    }
                )
    elif isinstance(payload, list):
        for item in payload:
            records.extend(_coerce_importance_records(item))
    return records


def _extract_features_used(*payloads: Dict[str, Any]) -> List[str]:
    features: List[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ("features_used", "feature_names", "model_features", "features"):
            raw = payload.get(key)
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item.strip():
                        features.append(item.strip())
                    elif isinstance(item, dict):
                        name = item.get("feature") or item.get("column") or item.get("name")
                        if name:
                            features.append(str(name))
    return list(dict.fromkeys(features))


def _normalize_importances(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    merged: Dict[str, float] = defaultdict(float)
    for item in records:
        feature = str(item.get("feature") or "").strip()
        value = _safe_float(item.get("importance"))
        if not feature or value is None:
            continue
        merged[feature] += abs(float(value))
    total = float(sum(merged.values()))
    normalized = []
    for feature, value in merged.items():
        normalized.append(
            {
                "feature": feature,
                "importance_abs": round(float(value), 10),
                "share": round(float(value) / total, 6) if total > 0 else None,
                "source_family": _infer_source_family(feature),
            }
        )
    normalized.sort(key=lambda item: float(item.get("importance_abs") or 0.0), reverse=True)
    return normalized, total


def _family_shares(importances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_totals: Dict[str, float] = defaultdict(float)
    family_features: Dict[str, List[str]] = defaultdict(list)
    total = 0.0
    for item in importances:
        value = _safe_float(item.get("importance_abs")) or 0.0
        family = str(item.get("source_family") or "unknown_or_domain_specific")
        total += value
        family_totals[family] += value
        family_features[family].append(str(item.get("feature") or ""))
    out = []
    for family, value in family_totals.items():
        out.append(
            {
                "family": family,
                "importance_abs": round(value, 10),
                "share": round(value / total, 6) if total > 0 else None,
                "features_sample": family_features[family][:12],
                "feature_count": len(family_features[family]),
            }
        )
    out.sort(key=lambda item: float(item.get("importance_abs") or 0.0), reverse=True)
    return out


def build_model_dependency_context_pack(
    *,
    model_card: Dict[str, Any] | None = None,
    metrics_payload: Dict[str, Any] | None = None,
    feature_governance_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model_card = model_card if isinstance(model_card, dict) else {}
    metrics_payload = metrics_payload if isinstance(metrics_payload, dict) else {}
    feature_governance_pack = feature_governance_pack if isinstance(feature_governance_pack, dict) else {}

    raw_importances = []
    raw_importances.extend(_coerce_importance_records(model_card))
    raw_importances.extend(_coerce_importance_records(metrics_payload))
    importances, total_importance = _normalize_importances(raw_importances)
    family_shares = _family_shares(importances)
    features_used = _extract_features_used(model_card, metrics_payload)
    if not features_used and importances:
        features_used = [str(item.get("feature")) for item in importances if item.get("feature")]

    top1_share = importances[0].get("share") if importances else None
    top3_share = round(sum(float(item.get("share") or 0.0) for item in importances[:3]), 6) if importances else None
    top5_share = round(sum(float(item.get("share") or 0.0) for item in importances[:5]), 6) if importances else None
    family_top_share = family_shares[0].get("share") if family_shares else None

    feature_gov_signals = (
        feature_governance_pack.get("feature_governance_signals")
        if isinstance(feature_governance_pack.get("feature_governance_signals"), dict)
        else {}
    )
    duplicate_groups = feature_gov_signals.get("semantic_duplicate_groups") if isinstance(feature_gov_signals, dict) else []
    corr_pairs = feature_gov_signals.get("high_correlation_pairs") if isinstance(feature_gov_signals, dict) else []

    observations: List[str] = []
    if importances:
        observations.append("Feature importance evidence is available; assess whether dominant features/families create operational dependency risk.")
    else:
        observations.append("No model feature-importance evidence was found; reviewers should ask whether interpretability artifacts are required for this business case.")
    if duplicate_groups:
        observations.append("Pre-model feature governance found semantically related columns; compare final top features against those groups.")
    if corr_pairs:
        observations.append("Pre-model feature governance found highly correlated pairs; check whether final top features over-represent one concept.")
    if family_shares:
        observations.append("Family-level concentration is measured from feature names; use it as a prompt for senior interpretation, not as automatic rejection.")

    return {
        "schema_version": "1.0",
        "role": "advisory_context_only",
        "deterministic_policy": (
            "This pack measures model dependency evidence. It must not reject, approve, or override "
            "an agent decision by itself; reviewers and translators use it to reason."
        ),
        "source": {
            "model_card_present": bool(model_card),
            "metrics_payload_present": bool(metrics_payload),
            "feature_governance_pack_present": bool(feature_governance_pack),
            "importance_records_found": len(raw_importances),
            "features_used_count": len(features_used),
        },
        "model_dependency_signals": {
            "features_used_sample": features_used[:80],
            "feature_importance_available": bool(importances),
            "feature_importance_total_abs": round(total_importance, 10),
            "top_features": importances[:30],
            "top_feature_share": top1_share,
            "top_3_feature_share": top3_share,
            "top_5_feature_share": top5_share,
            "source_family_shares": family_shares[:20],
            "top_source_family_share": family_top_share,
            "pre_model_semantic_duplicate_groups_count": len(duplicate_groups) if isinstance(duplicate_groups, list) else 0,
            "pre_model_high_correlation_pairs_count": len(corr_pairs) if isinstance(corr_pairs, list) else 0,
        },
        "senior_reasoning_questions": [
            "Is the model overly dependent on one feature, one source family, or one upstream provider from an operational-risk perspective?",
            "Do dominant features duplicate the same business concept through multiple encodings?",
            "If a public/legal/third-party-like family dominates, what happens when that source is delayed, unavailable, costly, or distribution-shifted?",
            "Does the executive report distinguish statistical performance from operational robustness and source dependency?",
        ],
        "observations": observations,
        "agent_usage_guidance": {
            "ml_engineer": "Emit model_card/metrics with features_used and feature importances whenever compatible with the model family.",
            "qa_reviewer": "Use this as evidence to assess whether feature dependency risks were acknowledged; do not fail solely from concentration.",
            "review_board": "Treat unresolved dependency risk as a business caveat unless it contradicts a hard contract requirement.",
            "business_translator": "Translate feature/family dependency into operational risks, integration caveats, and next validation steps.",
        },
    }


def summarize_model_dependency_context_pack(pack: Dict[str, Any] | None, *, max_lines: int = 80) -> str:
    if not isinstance(pack, dict) or not pack:
        return ""
    source = pack.get("source") if isinstance(pack.get("source"), dict) else {}
    signals = pack.get("model_dependency_signals") if isinstance(pack.get("model_dependency_signals"), dict) else {}
    lines = [
        "MODEL_DEPENDENCY_CONTEXT_PACK_SUMMARY:",
        "- role: advisory_context_only; model dependency evidence for senior LLM reasoning, not deterministic pass/fail gates",
        f"- model_card_present: {source.get('model_card_present')}; metrics_payload_present: {source.get('metrics_payload_present')}; importance_records_found: {source.get('importance_records_found')}; features_used_count: {source.get('features_used_count')}",
        f"- concentration: top_feature_share={signals.get('top_feature_share')}, top_3_feature_share={signals.get('top_3_feature_share')}, top_5_feature_share={signals.get('top_5_feature_share')}, top_source_family_share={signals.get('top_source_family_share')}",
    ]
    top_features = signals.get("top_features") if isinstance(signals.get("top_features"), list) else []
    if top_features:
        lines.append(f"- top_features_sample: {top_features[:10]}")
    family_shares = signals.get("source_family_shares") if isinstance(signals.get("source_family_shares"), list) else []
    if family_shares:
        lines.append(f"- source_family_shares_sample: {family_shares[:8]}")
    observations = pack.get("observations") if isinstance(pack.get("observations"), list) else []
    if observations:
        lines.append("- observations:")
        for item in observations[:8]:
            lines.append(f"  - {item}")
    questions = pack.get("senior_reasoning_questions") if isinstance(pack.get("senior_reasoning_questions"), list) else []
    if questions:
        lines.append("- senior_reasoning_questions:")
        for item in questions[:8]:
            lines.append(f"  - {item}")
    return "\n".join(lines[: max(1, int(max_lines))])
