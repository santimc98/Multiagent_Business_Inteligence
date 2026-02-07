from src.graph.graph import check_execution_planner_success, run_data_engineer


def test_check_execution_planner_success_fails_with_rejected_summary():
    state = {
        "execution_contract": {"contract_version": "4.1"},
        "execution_contract_diagnostics": {
            "summary": {"accepted": False},
        },
    }

    assert check_execution_planner_success(state) == "failed"


def test_check_execution_planner_success_passes_with_accepted_contract():
    state = {
        "execution_contract": {"contract_version": "4.1"},
        "execution_contract_diagnostics": {
            "validation": {"accepted": True, "status": "ok"},
            "summary": {"accepted": True},
        },
    }

    assert check_execution_planner_success(state) == "success"


def test_run_data_engineer_halts_when_planner_diagnostics_reject_contract():
    state = {
        "run_id": None,
        "execution_contract": {"contract_version": "4.1"},
        "execution_contract_diagnostics": {
            "validation": {"accepted": False, "status": "error"},
            "summary": {"accepted": False},
        },
        "budget_counters": {},
    }

    result = run_data_engineer(state)

    assert result.get("pipeline_aborted_reason") == "execution_contract_invalid"
    assert result.get("data_engineer_failed") is True
