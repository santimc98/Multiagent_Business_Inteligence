from src.graph.graph import _sanitize_stale_runtime_packet_for_current_success


def test_stale_runtime_packet_is_downgraded_when_current_attempt_succeeded():
    packet = {
        "status": "NEEDS_IMPROVEMENT",
        "feedback": "Runtime failure: ValueError object columns Sector and Segmento.",
        "failed_gates": ["runtime_failure"],
        "hard_failures": ["runtime_failure"],
        "required_fixes": ["Fix runtime dtype error."],
    }

    sanitized = _sanitize_stale_runtime_packet_for_current_success(
        packet,
        runtime_ok=True,
        output_contract_report={"overall_status": "warning", "missing": []},
        performance_gaps=[],
    )

    assert sanitized["status"] == "APPROVE_WITH_WARNINGS"
    assert sanitized["failed_gates"] == []
    assert sanitized["hard_failures"] == []
    assert sanitized["required_fixes"] == []
    assert sanitized["stale_runtime_feedback_pruned"] is True
