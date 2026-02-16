from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.steps.retry_policy import should_reselect_strategy_on_retry


def _route_from_policy(can_reselect: bool) -> str:
    return "replan" if can_reselect else "engineer"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _case_compliance_missing_outputs() -> None:
    state = {
        "last_iteration_type": "compliance",
        "output_contract_report": {
            "overall_status": "error",
            "missing": ["data/submission.csv"],
        },
    }
    strategies = [
        {"title": "strategy_a"},
        {"title": "strategy_b"},
    ]
    selected = {"title": "strategy_a"}
    can_reselect, reason = should_reselect_strategy_on_retry(state, strategies, selected)
    route = _route_from_policy(can_reselect)
    _assert(can_reselect is False, f"Expected can_reselect=False, got {can_reselect} ({reason})")
    _assert(route == "engineer", f"Expected route=engineer, got {route} ({reason})")


def _case_invalid_required_columns() -> None:
    state = {
        "column_validation": {
            "status": "invalid_required_columns",
            "invalid_details": [
                {
                    "strategy_index": 0,
                    "strategy_title": "strategy_a",
                    "invalid_columns": ["unknown_feature"],
                }
            ],
            "over_budget_details": [],
        },
        "last_iteration_type": "metric",
    }
    strategies = [
        {"title": "strategy_a"},
        {"title": "strategy_b"},
    ]
    selected = {"title": "strategy_a"}
    can_reselect, reason = should_reselect_strategy_on_retry(state, strategies, selected)
    route = _route_from_policy(can_reselect)
    _assert(can_reselect is True, f"Expected can_reselect=True, got {can_reselect} ({reason})")
    _assert(route == "replan", f"Expected route=replan, got {route} ({reason})")


def _case_mixed_planning_blocker_plus_missing_outputs_priority() -> None:
    base_state = {
        "column_validation": {
            "status": "invalid_required_columns",
            "invalid_details": [
                {
                    "strategy_index": 0,
                    "strategy_title": "strategy_a",
                    "invalid_columns": ["unknown_feature"],
                }
            ],
            "over_budget_details": [],
        },
        "output_contract_report": {
            "overall_status": "error",
            "missing": ["data/submission.csv"],
        },
        "last_iteration_type": "compliance",
    }
    strategies = [
        {"title": "strategy_a"},
        {"title": "strategy_b"},
    ]
    selected = {"title": "strategy_a"}

    can_reselect, reason = should_reselect_strategy_on_retry(base_state, strategies, selected)
    route = _route_from_policy(can_reselect)
    _assert(can_reselect is True, f"Expected can_reselect=True in mixed planning+missing case, got {can_reselect} ({reason})")
    _assert(route == "replan", f"Expected route=replan in mixed planning+missing case, got {route} ({reason})")

    state_with_preflight_fail = dict(base_state)
    state_with_preflight_fail["last_gate_context"] = {"failed_gates": ["preflight_gate_B"]}
    can_reselect_fail, reason_fail = should_reselect_strategy_on_retry(
        state_with_preflight_fail,
        strategies,
        selected,
    )
    route_fail = _route_from_policy(can_reselect_fail)
    _assert(can_reselect_fail is False, f"Expected can_reselect=False when preflight FAIL exists, got {can_reselect_fail} ({reason_fail})")
    _assert(route_fail == "engineer", f"Expected route=engineer when preflight FAIL exists, got {route_fail} ({reason_fail})")


def main() -> int:
    _case_compliance_missing_outputs()
    _case_invalid_required_columns()
    _case_mixed_planning_blocker_plus_missing_outputs_priority()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
