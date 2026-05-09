from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compact_list(values: Any, limit: int = 8) -> List[Any]:
    if not isinstance(values, list):
        return []
    return values[: max(0, int(limit))]


def _extract_metric_records(metrics_payload: Dict[str, Any], run_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    def _add(name: Any, value: Any, source: str) -> None:
        if not str(name or "").strip():
            return
        records.append({"metric": str(name), "value": value, "source": source})

    if isinstance(metrics_payload, dict):
        for key, value in metrics_payload.items():
            if isinstance(value, (int, float, str, bool)) and any(
                token in str(key).lower()
                for token in ("auc", "accuracy", "mae", "rmse", "loss", "kappa", "qwk", "latency", "memory", "size", "deviation", "coverage")
            ):
                _add(key, value, "metrics_payload")
        model_perf = metrics_payload.get("model_performance")
        if isinstance(model_perf, dict):
            for key, value in model_perf.items():
                if isinstance(value, (int, float, str, bool)):
                    _add(key, value, "metrics_payload.model_performance")
    if isinstance(run_summary, dict):
        run_metrics = run_summary.get("metrics")
        if isinstance(run_metrics, dict):
            for key, value in run_metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    _add(key, value, "run_summary.metrics")
    seen = set()
    unique = []
    for item in records:
        key = (str(item.get("metric")), str(item.get("source")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:40]


def build_business_impact_context_pack(
    *,
    business_objective: str = "",
    executive_decision_label: str = "",
    run_summary: Dict[str, Any] | None = None,
    metrics_payload: Dict[str, Any] | None = None,
    data_adequacy_report: Dict[str, Any] | None = None,
    case_alignment_report: Dict[str, Any] | None = None,
    integration_card: Dict[str, Any] | None = None,
    model_dependency_context_pack: Dict[str, Any] | None = None,
    feature_governance_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_summary = run_summary if isinstance(run_summary, dict) else {}
    metrics_payload = metrics_payload if isinstance(metrics_payload, dict) else {}
    data_adequacy_report = data_adequacy_report if isinstance(data_adequacy_report, dict) else {}
    case_alignment_report = case_alignment_report if isinstance(case_alignment_report, dict) else {}
    integration_card = integration_card if isinstance(integration_card, dict) else {}
    model_dependency_context_pack = (
        model_dependency_context_pack if isinstance(model_dependency_context_pack, dict) else {}
    )
    feature_governance_pack = feature_governance_pack if isinstance(feature_governance_pack, dict) else {}

    metric_records = _extract_metric_records(metrics_payload, run_summary)
    data_status = str(data_adequacy_report.get("status") or "").strip()
    case_status = str(case_alignment_report.get("status") or "").strip()
    integration_output = integration_card.get("output_contract") if isinstance(integration_card.get("output_contract"), dict) else {}
    integration_exec = integration_card.get("execution_contract") if isinstance(integration_card.get("execution_contract"), dict) else {}
    model_dep_signals = (
        model_dependency_context_pack.get("model_dependency_signals")
        if isinstance(model_dependency_context_pack.get("model_dependency_signals"), dict)
        else {}
    )
    feature_gov_signals = (
        feature_governance_pack.get("feature_governance_signals")
        if isinstance(feature_governance_pack.get("feature_governance_signals"), dict)
        else {}
    )

    caveats: List[Dict[str, Any]] = []
    if data_status and data_status.upper() not in {"OK", "PASS", "GO", "SUFFICIENT"}:
        caveats.append(
            {
                "area": "data_adequacy",
                "status": data_status,
                "evidence": _compact_list(data_adequacy_report.get("reasons"), 5),
                "business_question": "Does the available data support the intended operational decision, or should the model be piloted with limitations?",
            }
        )
    if case_status and case_status.upper() not in {"PASS", "OK"}:
        caveats.append(
            {
                "area": "case_alignment",
                "status": case_status,
                "evidence": _compact_list(case_alignment_report.get("failures"), 6),
                "business_question": "Which business cases or segments are not aligned with the intended decision policy?",
            }
        )
    missing_artifacts = integration_output.get("missing_required_artifacts")
    if isinstance(missing_artifacts, list) and missing_artifacts:
        caveats.append(
            {
                "area": "integration_readiness",
                "status": "missing_required_artifacts",
                "evidence": missing_artifacts[:8],
                "business_question": "Can the result be handed off to engineering without manual reconstruction?",
            }
        )
    if model_dep_signals.get("top_feature_share") is not None or model_dep_signals.get("top_source_family_share") is not None:
        caveats.append(
            {
                "area": "model_dependency",
                "status": "dependency_evidence_available",
                "evidence": {
                    "top_feature_share": model_dep_signals.get("top_feature_share"),
                    "top_source_family_share": model_dep_signals.get("top_source_family_share"),
                    "source_family_shares": _compact_list(model_dep_signals.get("source_family_shares"), 5),
                },
                "business_question": "If dominant variables or source families change, become unavailable, or shift distribution, how resilient is the model?",
            }
        )
    duplicate_groups = feature_gov_signals.get("semantic_duplicate_groups")
    if isinstance(duplicate_groups, list) and duplicate_groups:
        caveats.append(
            {
                "area": "feature_governance",
                "status": "semantic_overlap_evidence_available",
                "evidence": duplicate_groups[:5],
                "business_question": "Are duplicated business concepts over-weighting a factor that should be represented once or explicitly justified?",
            }
        )

    operational_examples = []
    decisioning = integration_card.get("business_context") if isinstance(integration_card.get("business_context"), dict) else {}
    decisioning_req = decisioning.get("decisioning_requirements") if isinstance(decisioning.get("decisioning_requirements"), dict) else {}
    if decisioning_req:
        operational_examples.append(
            {
                "type": "decisioning_policy",
                "evidence": decisioning_req,
                "interpretation_prompt": "Explain how model outputs would be consumed by the business decision workflow, without inventing thresholds not present in the contract.",
            }
        )
    if metric_records:
        operational_examples.append(
            {
                "type": "metric_translation",
                "evidence": metric_records[:10],
                "interpretation_prompt": "Translate metric values into business meaning, limits, and next validation actions.",
            }
        )

    return {
        "schema_version": "1.0",
        "role": "business_impact_advisory",
        "deterministic_policy": (
            "This pack organizes business-impact evidence and questions. It must not invent decisions, "
            "thresholds, or deployment policy; the LLM must reason from the cited facts."
        ),
        "decision_context": {
            "business_objective_excerpt": str(business_objective or "")[:2500],
            "executive_decision_label": executive_decision_label,
            "run_outcome": run_summary.get("run_outcome"),
            "run_status": run_summary.get("status"),
            "failed_gates": _compact_list(run_summary.get("failed_gates"), 12),
            "warnings": _compact_list(run_summary.get("warnings"), 12),
        },
        "metric_context": {
            "records": metric_records,
            "primary_metric": metrics_payload.get("primary_metric") or run_summary.get("primary_metric"),
        },
        "business_caveats": caveats,
        "operational_examples": operational_examples,
        "integration_readiness": {
            "entrypoints": _compact_list(integration_exec.get("entrypoints"), 6),
            "model_artifact_paths": _compact_list(integration_exec.get("model_artifact_paths"), 8),
            "missing_required_artifacts": missing_artifacts if isinstance(missing_artifacts, list) else [],
            "runtime_requirements": integration_exec.get("runtime_requirements") if isinstance(integration_exec, dict) else {},
        },
        "senior_reasoning_questions": [
            "What does the measured model performance mean for the concrete business workflow, not just statistically?",
            "Which limitations are technical, which are data limitations, and which are operational or governance risks?",
            "What should a human decision-maker do next: pilot, monitor, collect data, simplify integration, or reject?",
            "Can the report cite evidence for every operational claim and avoid deployment promises not present in artifacts?",
        ],
    }


def summarize_business_impact_context_pack(pack: Dict[str, Any] | None, *, max_lines: int = 80) -> str:
    if not isinstance(pack, dict) or not pack:
        return ""
    decision = pack.get("decision_context") if isinstance(pack.get("decision_context"), dict) else {}
    metrics = pack.get("metric_context") if isinstance(pack.get("metric_context"), dict) else {}
    caveats = pack.get("business_caveats") if isinstance(pack.get("business_caveats"), list) else []
    integration = pack.get("integration_readiness") if isinstance(pack.get("integration_readiness"), dict) else {}
    examples = pack.get("operational_examples") if isinstance(pack.get("operational_examples"), list) else []
    lines = [
        "BUSINESS_IMPACT_CONTEXT_PACK_SUMMARY:",
        "- role: business_impact_advisory; operational interpretation facts/questions, not deterministic deployment policy",
        f"- decision: label={decision.get('executive_decision_label')}, run_outcome={decision.get('run_outcome')}, run_status={decision.get('run_status')}",
        f"- metric_records_count: {len(metrics.get('records') or [])}; primary_metric: {metrics.get('primary_metric')}",
        f"- caveats_count: {len(caveats)}; integration_missing_required_artifacts: {integration.get('missing_required_artifacts')}",
    ]
    if caveats:
        lines.append("- caveats_sample:")
        for item in caveats[:8]:
            lines.append(
                "  "
                + str(
                    {
                        "area": item.get("area"),
                        "status": item.get("status"),
                        "business_question": item.get("business_question"),
                    }
                )
            )
    if examples:
        lines.append(f"- operational_examples_sample: {examples[:4]}")
    questions = pack.get("senior_reasoning_questions") if isinstance(pack.get("senior_reasoning_questions"), list) else []
    if questions:
        lines.append("- senior_reasoning_questions:")
        for question in questions[:8]:
            lines.append(f"  - {question}")
    return "\n".join(lines[: max(1, int(max_lines))])
