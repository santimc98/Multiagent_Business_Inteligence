"""
Result evaluator utilities extracted from graph.py (seniority refactoring).

Functions for review packet harmonization and consistency guards.
"""

from typing import Any, Dict, List


def _looks_blocking_retry_signal(text: Any) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    blocking_tokens = [
        "runtime", "traceback", "exception", "crash", "sandbox",
        "output_contract", "contract_required", "required_outputs_missing",
        "artifact_missing", "missing output", "missing artifact",
        "qa_", "code_audit", "hard_fail", "leakage", "security",
        "decisioning", "visual_requirements", "schema_inconsistent",
        "alignment_check", "segmentation", "operational_logic",
        "decisioning", "uncertainty", "dependency", "dialect",
        "static_precheck", "syntax",
    ]
    return any(token in value for token in blocking_tokens)


def _apply_review_consistency_guard(
    result: Dict[str, Any] | None,
    diagnostics: Dict[str, Any] | None,
    *,
    actor: str,
) -> Dict[str, Any]:
    packet: Dict[str, Any] = dict(result or {})
    blockers = [str(x) for x in ((diagnostics or {}).get("hard_blockers") or []) if x]
    if not blockers:
        return packet
    status = str(packet.get("status") or "").upper()
    if status and status not in {"APPROVED", "APPROVE_WITH_WARNINGS"}:
        return packet

    actor_name = str(actor or "reviewer").strip().lower() or "reviewer"
    blocker_text = ", ".join(blockers)
    note = (
        f"{actor_name.upper()}_CONTEXT_GUARD: forced REJECTED due to deterministic blockers "
        f"({blocker_text})."
    )
    packet["status"] = "REJECTED"
    feedback = str(packet.get("feedback") or "").strip()
    packet["feedback"] = f"{feedback}\n{note}".strip() if feedback else note

    failed_gates = packet.get("failed_gates")
    if not isinstance(failed_gates, list):
        failed_gates = []
    for blocker in blockers:
        if blocker not in failed_gates:
            failed_gates.append(blocker)
    packet["failed_gates"] = failed_gates

    hard_failures = packet.get("hard_failures")
    if not isinstance(hard_failures, list):
        hard_failures = []
    for blocker in blockers:
        if blocker not in hard_failures:
            hard_failures.append(blocker)
    packet["hard_failures"] = hard_failures

    required_fixes = packet.get("required_fixes")
    if not isinstance(required_fixes, list):
        required_fixes = []
    generic_fix = "Resolve deterministic runtime/output blockers before requesting approval again."
    if generic_fix not in required_fixes:
        required_fixes.append(generic_fix)
    packet["required_fixes"] = required_fixes
    return packet


def _harmonize_review_packets_with_final_eval(
    reviewer_packet: Dict[str, Any] | None,
    qa_packet: Dict[str, Any] | None,
    *,
    eval_raw_status: str,
    eval_feedback: str,
    eval_failed_gates: List[str] | None,
    eval_required_fixes: List[str] | None,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Keep reviewer/QA packets semantically aligned with the final evaluator signal.
    This does not force deterministic rejection; it only downgrades to warnings
    when evaluator reports unresolved issues but packets are clean approvals.
    """
    trigger_statuses = {"NEEDS_IMPROVEMENT", "REJECTED", "FAIL", "FAILED", "ERROR", "CRASH"}
    status_norm = str(eval_raw_status or "").strip().upper()
    if status_norm not in trigger_statuses:
        return dict(reviewer_packet or {}), dict(qa_packet or {}), []

    advisory_fixes = [str(item) for item in (eval_required_fixes or []) if item][:5]
    advisory_gates = [str(item) for item in (eval_failed_gates or []) if item][:5]
    eval_feedback_trimmed = str(eval_feedback or "").strip()
    blocking_eval = any(_looks_blocking_retry_signal(item) for item in (advisory_gates + advisory_fixes))
    if not blocking_eval and eval_feedback_trimmed:
        blocking_eval = _looks_blocking_retry_signal(eval_feedback_trimmed)
    notes: List[str] = []

    def _patch_packet(packet: Dict[str, Any], label: str) -> Dict[str, Any]:
        patched = dict(packet or {})
        status = str(patched.get("status") or "").strip().upper()
        failed = [str(item) for item in (patched.get("failed_gates") or []) if item]
        hard = [str(item) for item in (patched.get("hard_failures") or []) if item]
        if status != "APPROVED" or failed or hard:
            return patched

        if blocking_eval:
            patched["status"] = "REJECTED"
            message = (
                f"{label.upper()}_CROSS_REVIEW_ALIGNMENT: deterministic/blocking issues reported by "
                "result_evaluator; approval converted to REJECTED."
            )
        else:
            patched["status"] = "APPROVE_WITH_WARNINGS"
            message = (
                f"{label.upper()}_CROSS_REVIEW_ALIGNMENT: result_evaluator reported unresolved "
                "contract/business coverage issues; approval downgraded to warning-only context."
            )
        feedback = str(patched.get("feedback") or "").strip()
        if eval_feedback_trimmed:
            message = f"{message} evaluator_feedback={eval_feedback_trimmed[:320]}"
        patched["feedback"] = f"{feedback}\n{message}".strip() if feedback else message

        if "cross_review_alignment_gap" not in failed:
            failed.append("cross_review_alignment_gap")
        for gate in advisory_gates:
            if gate not in failed:
                failed.append(gate)
        patched["failed_gates"] = failed

        required = [str(item) for item in (patched.get("required_fixes") or []) if item]
        for fix in advisory_fixes:
            if fix not in required:
                required.append(fix)
        if blocking_eval and not required:
            required.append("Resolve deterministic blockers before requesting approval again.")
        if required:
            patched["required_fixes"] = required
        if blocking_eval:
            hard_failures = [str(item) for item in (patched.get("hard_failures") or []) if item]
            for blocker in advisory_gates:
                if _looks_blocking_retry_signal(blocker) and blocker not in hard_failures:
                    hard_failures.append(blocker)
            if "cross_review_alignment_gap" not in hard_failures:
                hard_failures.append("cross_review_alignment_gap")
            patched["hard_failures"] = hard_failures
            notes.append(f"{label}: escalated APPROVED -> REJECTED due to blocking evaluator status {status_norm}")
        else:
            notes.append(f"{label}: downgraded APPROVED -> APPROVE_WITH_WARNINGS due to evaluator status {status_norm}")
        return patched

    return _patch_packet(dict(reviewer_packet or {}), "reviewer"), _patch_packet(dict(qa_packet or {}), "qa_reviewer"), notes
