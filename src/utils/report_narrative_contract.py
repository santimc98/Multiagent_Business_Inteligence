from __future__ import annotations

from typing import Any, Dict, List


def _has_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _has_nested(pack: Dict[str, Any], *path: str) -> bool:
    current: Any = pack
    for key in path:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return _has_payload(current)


def build_report_narrative_contract(
    *,
    data_quality_shape_pack: Dict[str, Any] | None = None,
    feature_governance_pack: Dict[str, Any] | None = None,
    model_dependency_context_pack: Dict[str, Any] | None = None,
    integration_card: Dict[str, Any] | None = None,
    business_impact_context_pack: Dict[str, Any] | None = None,
    qa_review_signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data_quality_shape_pack = data_quality_shape_pack if isinstance(data_quality_shape_pack, dict) else {}
    feature_governance_pack = feature_governance_pack if isinstance(feature_governance_pack, dict) else {}
    model_dependency_context_pack = (
        model_dependency_context_pack if isinstance(model_dependency_context_pack, dict) else {}
    )
    integration_card = integration_card if isinstance(integration_card, dict) else {}
    business_impact_context_pack = (
        business_impact_context_pack if isinstance(business_impact_context_pack, dict) else {}
    )
    qa_review_signals = qa_review_signals if isinstance(qa_review_signals, dict) else {}

    required_topics: List[Dict[str, Any]] = []

    if _has_nested(data_quality_shape_pack, "shape_signals", "counts"):
        required_topics.append(
            {
                "id": "data_quality_characterization",
                "source": "data_quality_shape_pack",
                "narrative_obligation": (
                    "If material, explain dispersion, missingness, zero-vs-null, concentration, "
                    "or low-variability facts and how they affect model trust or data readiness."
                ),
                "non_goal": "Do not turn advisory data-shape warnings into deterministic rejection.",
            }
        )
    if _has_nested(feature_governance_pack, "feature_governance_signals"):
        required_topics.append(
            {
                "id": "feature_governance",
                "source": "feature_governance_pack",
                "narrative_obligation": (
                    "Explain material duplicated concepts, correlated variables, feature-family balance, "
                    "or feature-selection governance risks when evidence exists."
                ),
                "non_goal": "Do not claim a variable should be dropped unless the run evidence supports that decision.",
            }
        )
    if _has_nested(model_dependency_context_pack, "model_dependency_signals"):
        required_topics.append(
            {
                "id": "model_dependency_and_dominance",
                "source": "model_dependency_context_pack",
                "narrative_obligation": (
                    "Translate feature/family dominance and source dependency into operational caveats "
                    "or monitoring recommendations when relevant."
                ),
                "non_goal": "Do not fail the model solely because concentration exists.",
            }
        )
    if integration_card:
        required_topics.append(
            {
                "id": "integration_readiness",
                "source": "integration_card",
                "narrative_obligation": (
                    "Include the production handoff facts that matter: inputs, scoring entrypoint, outputs, "
                    "runtime/model-size evidence, missing handoff artifacts, and integration caveats."
                ),
                "non_goal": "Do not invent APIs, thresholds, SLAs, or deployment policy not present in artifacts.",
            }
        )
    if _has_nested(business_impact_context_pack, "decision_context") or _has_nested(
        business_impact_context_pack, "business_caveats"
    ):
        required_topics.append(
            {
                "id": "business_impact_interpretation",
                "source": "business_impact_context_pack",
                "narrative_obligation": (
                    "Translate metrics and caveats into what the business can do next, separating facts, "
                    "cautious inference, and recommended action."
                ),
                "non_goal": "Do not present recommendations as established run facts.",
            }
        )
    if qa_review_signals.get("status_is_warning") or qa_review_signals.get("explicit_warnings") or qa_review_signals.get("soft_failures"):
        required_topics.append(
            {
                "id": "qa_material_caveats",
                "source": "qa_review_signals",
                "narrative_obligation": "Surface material QA warnings as limitations or confidence caveats.",
                "non_goal": "Do not let a positive executive outcome erase reviewer caveats.",
            }
        )

    return {
        "schema_version": "1.0",
        "role": "translator_narrative_contract",
        "deterministic_policy": (
            "This contract governs report coverage and evidence discipline. It does not decide the business outcome."
        ),
        "required_topics": required_topics,
        "global_rules": [
            "Make the authoritative executive decision and rationale clear early.",
            "Use evidence-backed claims; mark unsupported interpretations as cautious inference.",
            "Skip topics only when the corresponding evidence is unavailable or immaterial, and do not fabricate detail.",
            "Inline artifacts next to the claims they support; do not create a generic visual-analysis section.",
            "Separate final incumbent facts from rejected experiments or historical reviewer states.",
        ],
    }


def summarize_report_narrative_contract(contract: Dict[str, Any] | None) -> str:
    if not isinstance(contract, dict) or not contract:
        return ""
    topics = contract.get("required_topics") if isinstance(contract.get("required_topics"), list) else []
    lines = [
        "REPORT_NARRATIVE_CONTRACT:",
        "- role: translator_narrative_contract; report coverage/evidence discipline, not outcome decision logic",
        f"- required_topics: {[str(item.get('id')) for item in topics if isinstance(item, dict)]}",
    ]
    for item in topics[:10]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "  "
            + str(
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "obligation": item.get("narrative_obligation"),
                    "non_goal": item.get("non_goal"),
                }
            )
        )
    return "\n".join(lines)
