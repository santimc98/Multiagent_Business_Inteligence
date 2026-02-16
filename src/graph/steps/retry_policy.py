from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _safe_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_strategy_title(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_runtime_text(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("last_runtime_error_tail", "execution_output", "error_message"):
        value = state.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts).lower()


def _has_runtime_failure_marker(state: Dict[str, Any]) -> bool:
    runtime_text = _extract_runtime_text(state)
    if not runtime_text:
        return False
    tokens = [
        "traceback (most recent call last)",
        "execution error",
        "failed_runtime",
        "sandbox execution failed",
        "runtimeerror",
        "modulenotfounderror",
        "importerror",
        "nameerror",
        "typeerror",
        "valueerror",
        "keyerror",
        "indexerror",
        "attributeerror",
        "syntaxerror",
        "permissionerror",
        "memoryerror",
        "killed",
    ]
    return any(token in runtime_text for token in tokens)


def _contains_compliance_or_safety_signal(state: Dict[str, Any], gate_context: Dict[str, Any]) -> bool:
    signal_tokens = (
        "security",
        "safety",
        "forbidden",
        "blocked",
        "code_audit",
        "leakage",
    )
    signals: List[str] = []
    for key in ("failed_gates", "required_fixes", "hard_failures"):
        values = gate_context.get(key)
        if isinstance(values, list):
            signals.extend(str(item).lower() for item in values if item)
    for key in ("review_feedback", "execution_feedback", "error_message"):
        value = state.get(key)
        if value:
            signals.append(str(value).lower())
    joined = " ".join(signals)
    return any(token in joined for token in signal_tokens)


def _has_preflight_fail_signal(state: Dict[str, Any], gate_context: Dict[str, Any]) -> bool:
    for key in ("failed_gates", "required_fixes", "hard_failures"):
        values = gate_context.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value or "").strip().lower()
            if not text:
                continue
            if "preflight_gate_" in text or "pre_flight_gates" in text:
                return True
            if "preflight" in text and "fail" in text:
                return True

    iteration_handoff = state.get("iteration_handoff") if isinstance(state.get("iteration_handoff"), dict) else {}
    preflight_gates = iteration_handoff.get("preflight_gates") if isinstance(iteration_handoff.get("preflight_gates"), dict) else {}
    if isinstance(preflight_gates.get("fails"), list) and any(preflight_gates.get("fails")):
        return True

    execution_output = str(state.get("execution_output") or "")
    if execution_output and "pre_flight_gates" in execution_output.lower():
        if re.search(r"gate\s+[a-z0-9]+\s*:\s*fail\b", execution_output, flags=re.IGNORECASE):
            return True
    return False


def _detect_engineering_blocker(
    state: Dict[str, Any],
    *,
    allow_planning_replan_on_missing_outputs: bool = False,
) -> str:
    gate_context = state.get("last_gate_context") if isinstance(state.get("last_gate_context"), dict) else {}

    if state.get("runtime_fix_terminal"):
        return "runtime_fix_terminal"
    if state.get("sandbox_failed"):
        return "sandbox_failed"
    if _has_runtime_failure_marker(state):
        return "runtime_failure_marker"
    if _has_preflight_fail_signal(state, gate_context):
        return "preflight_gate_fail"
    if _contains_compliance_or_safety_signal(state, gate_context):
        return "compliance_or_safety_signal"

    last_iter_type = _safe_lower(state.get("last_iteration_type"))
    gate_iter_type = _safe_lower(gate_context.get("iteration_type"))
    if (last_iter_type == "compliance" or gate_iter_type == "compliance") and not allow_planning_replan_on_missing_outputs:
        return "compliance_iteration"

    oc_report = state.get("output_contract_report") if isinstance(state.get("output_contract_report"), dict) else {}
    if isinstance(oc_report, dict):
        missing = [str(item) for item in (oc_report.get("missing") or []) if item]
        if missing and not allow_planning_replan_on_missing_outputs:
            return "missing_required_outputs"
        overall_status = _safe_lower(oc_report.get("overall_status"))
        if overall_status == "error" and not (allow_planning_replan_on_missing_outputs and missing):
            return "output_contract_error"
        artifact_report = oc_report.get("artifact_requirements_report")
        artifact_status = _safe_lower(artifact_report.get("status")) if isinstance(artifact_report, dict) else ""
        if artifact_status == "error" and not (allow_planning_replan_on_missing_outputs and missing):
            return "artifact_requirements_error"

    return ""


def _resolve_selected_strategy_index(
    strategies_list: List[Dict[str, Any]],
    selected_strategy: Dict[str, Any] | None,
    selected_index_hint: Any,
) -> int | None:
    if not isinstance(strategies_list, list) or not strategies_list:
        return None

    idx_hint = _safe_int(selected_index_hint)
    if idx_hint is not None and 0 <= idx_hint < len(strategies_list):
        return idx_hint

    if not isinstance(selected_strategy, dict):
        return None

    for key in ("strategy_index", "_strategy_index", "index"):
        idx = _safe_int(selected_strategy.get(key))
        if idx is not None and 0 <= idx < len(strategies_list):
            return idx

    selected_title = _normalize_strategy_title(selected_strategy.get("title"))
    if not selected_title:
        return None
    for idx, strategy in enumerate(strategies_list):
        if not isinstance(strategy, dict):
            continue
        if _normalize_strategy_title(strategy.get("title")) == selected_title:
            return idx
    return None


def _detail_matches_selected(
    detail: Dict[str, Any],
    selected_index: int | None,
    selected_title_norm: str,
) -> bool:
    detail_index = _safe_int(detail.get("strategy_index"))
    if detail_index is not None and selected_index is not None and detail_index == selected_index:
        return True

    detail_title = (
        detail.get("strategy_title")
        or detail.get("title")
        or detail.get("strategy_name")
        or detail.get("name")
    )
    detail_title_norm = _normalize_strategy_title(detail_title)
    if selected_title_norm and detail_title_norm and detail_title_norm == selected_title_norm:
        return True

    return False


def _is_strategy_planning_blocked(
    index: int,
    strategy: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> bool:
    title_norm = _normalize_strategy_title(strategy.get("title"))
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_index = _safe_int(detail.get("strategy_index"))
        if detail_index is not None and detail_index == index:
            return True
        detail_title = (
            detail.get("strategy_title")
            or detail.get("title")
            or detail.get("strategy_name")
            or detail.get("name")
        )
        detail_title_norm = _normalize_strategy_title(detail_title)
        if title_norm and detail_title_norm and detail_title_norm == title_norm:
            return True
    return False


def should_reselect_strategy_on_retry(
    state: Dict[str, Any],
    strategies_list: List[Dict[str, Any]],
    selected_strategy: Dict[str, Any] | None,
) -> Tuple[bool, str]:
    """
    Decide if retry should reselect strategy (replan) instead of staying on engineer.

    Universal policy:
    - Engineering/compliance/runtime/safety blockers => stay on engineer.
    - Reselect only for planning-level strategy infeasibility from column_validation.
    """
    state = state if isinstance(state, dict) else {}
    strategies_list = strategies_list if isinstance(strategies_list, list) else []
    selected_strategy = selected_strategy if isinstance(selected_strategy, dict) else None

    if len(strategies_list) <= 1:
        return False, "single_strategy_or_missing"

    column_validation = state.get("column_validation") if isinstance(state.get("column_validation"), dict) else {}
    status = _safe_lower(column_validation.get("status"))
    planning_statuses = {"invalid_required_columns", "required_columns_over_budget"}
    planning_blocker_declared = status in planning_statuses

    blocker_reason = _detect_engineering_blocker(
        state,
        allow_planning_replan_on_missing_outputs=planning_blocker_declared,
    )
    if blocker_reason:
        return False, blocker_reason

    if status not in planning_statuses:
        return False, f"column_validation_status:{status or 'none'}"

    invalid_details = column_validation.get("invalid_details")
    over_budget_details = column_validation.get("over_budget_details")
    invalid_details = invalid_details if isinstance(invalid_details, list) else []
    over_budget_details = over_budget_details if isinstance(over_budget_details, list) else []
    planning_details: List[Dict[str, Any]] = [
        detail for detail in (invalid_details + over_budget_details) if isinstance(detail, dict)
    ]
    if not planning_details:
        return False, "planning_blocker_details_missing"

    selected_index = _resolve_selected_strategy_index(
        strategies_list,
        selected_strategy,
        state.get("selected_strategy_index"),
    )
    selected_title_norm = _normalize_strategy_title(
        selected_strategy.get("title") if isinstance(selected_strategy, dict) else ""
    )
    selected_blocked = any(
        _detail_matches_selected(detail, selected_index, selected_title_norm)
        for detail in planning_details
    )
    if not selected_blocked:
        return False, "selected_strategy_not_planning_blocked"

    valid_strategy_count = len([s for s in strategies_list if isinstance(s, dict)])
    blocked_count = 0
    for idx, strategy in enumerate(strategies_list):
        if not isinstance(strategy, dict):
            continue
        if _is_strategy_planning_blocked(idx, strategy, planning_details):
            blocked_count += 1
    if valid_strategy_count and blocked_count >= valid_strategy_count:
        return False, "all_strategies_planning_blocked"

    return True, f"planning_blocker:{status}"
