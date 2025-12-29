import json
import os
from datetime import datetime
from typing import Any, Dict


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_governance_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    output_contract = _safe_load_json("data/output_contract_report.json")
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    integrity = _safe_load_json("data/integrity_audit_report.json")

    issues = integrity.get("issues", []) if isinstance(integrity, dict) else []
    severity_counts = {}
    for issue in issues:
        sev = str(issue.get("severity", "unknown"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    review_verdict = state.get("last_successful_review_verdict") or state.get("review_verdict")
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")

    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_title": contract.get("strategy_title", ""),
        "business_objective": contract.get("business_objective", ""),
        "review_verdict": review_verdict,
        "last_gate_context": gate_context,
        "output_contract": output_contract,
        "case_alignment": case_alignment,
        "alignment_check": alignment_check,
        "integrity_issues_summary": severity_counts,
        "budget_counters": state.get("budget_counters", {}),
        "run_budget": state.get("run_budget", {}),
        "data_risks": contract.get("data_risks", []),
    }


def write_governance_report(state: Dict[str, Any], path: str = "data/governance_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = build_governance_report(state)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def build_run_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    output_contract = _safe_load_json("data/output_contract_report.json")
    data_adequacy = _safe_load_json("data/data_adequacy_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    status = state.get("last_successful_review_verdict") or state.get("review_verdict") or "UNKNOWN"
    failed_gates = []
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")
    if isinstance(gate_context, dict):
        failed_gates = gate_context.get("failed_gates", []) or []
    if isinstance(case_alignment, dict) and case_alignment.get("status") == "FAIL":
        failed_gates.extend(case_alignment.get("failures", []))
    if isinstance(output_contract, dict) and output_contract.get("missing"):
        failed_gates.append("output_contract_missing")
    adequacy_summary = {}
    if isinstance(data_adequacy, dict):
        alignment = data_adequacy.get("quality_gates_alignment", {}) if isinstance(data_adequacy, dict) else {}
        alignment_summary = {}
        if isinstance(alignment, dict) and alignment:
            mapped = alignment.get("mapped_gates", {}) if isinstance(alignment, dict) else {}
            unmapped = alignment.get("unmapped_gates", {}) if isinstance(alignment, dict) else {}
            alignment_summary = {
                "status": alignment.get("status"),
                "mapped_gate_count": len(mapped) if isinstance(mapped, dict) else 0,
                "unmapped_gate_count": len(unmapped) if isinstance(unmapped, dict) else 0,
            }
        adequacy_summary = {
            "status": data_adequacy.get("status"),
            "reasons": data_adequacy.get("reasons", []),
            "recommendations": data_adequacy.get("recommendations", []),
            "consecutive_data_limited": data_adequacy.get("consecutive_data_limited"),
            "data_limited_threshold": data_adequacy.get("data_limited_threshold"),
            "threshold_reached": data_adequacy.get("threshold_reached"),
            "quality_gates_alignment": alignment_summary,
        }
    return {
        "run_id": state.get("run_id"),
        "status": status,
        "failed_gates": list(dict.fromkeys(failed_gates)),
        "budget_counters": state.get("budget_counters", {}),
        "data_adequacy": adequacy_summary,
        "alignment_check": {
            "status": alignment_check.get("status"),
            "failure_mode": alignment_check.get("failure_mode"),
            "summary": alignment_check.get("summary"),
        } if isinstance(alignment_check, dict) and alignment_check else {},
    }
