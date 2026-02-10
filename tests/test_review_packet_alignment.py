from src.graph import graph as graph_mod


def test_harmonize_review_packets_downgrades_clean_approvals_when_eval_needs_improvement():
    reviewer_packet = {"status": "APPROVED", "feedback": "Code quality good.", "failed_gates": [], "hard_failures": []}
    qa_packet = {"status": "APPROVED", "feedback": "QA passed.", "failed_gates": [], "hard_failures": []}

    reviewer_adj, qa_adj, notes = graph_mod._harmonize_review_packets_with_final_eval(
        reviewer_packet,
        qa_packet,
        eval_raw_status="NEEDS_IMPROVEMENT",
        eval_feedback="Missing business objective coverage in segmented reporting.",
        eval_failed_gates=["business_objective_coverage"],
        eval_required_fixes=["Add segmented report by required business dimensions."],
    )

    assert reviewer_adj.get("status") == "APPROVE_WITH_WARNINGS"
    assert qa_adj.get("status") == "APPROVE_WITH_WARNINGS"
    assert "cross_review_alignment_gap" in (reviewer_adj.get("failed_gates") or [])
    assert "cross_review_alignment_gap" in (qa_adj.get("failed_gates") or [])
    assert notes


def test_harmonize_review_packets_noop_when_eval_not_escalated():
    reviewer_packet = {"status": "APPROVED", "feedback": "ok", "failed_gates": [], "hard_failures": []}
    qa_packet = {"status": "APPROVE_WITH_WARNINGS", "feedback": "warn", "failed_gates": [], "hard_failures": []}

    reviewer_adj, qa_adj, notes = graph_mod._harmonize_review_packets_with_final_eval(
        reviewer_packet,
        qa_packet,
        eval_raw_status="APPROVE_WITH_WARNINGS",
        eval_feedback="",
        eval_failed_gates=[],
        eval_required_fixes=[],
    )

    assert reviewer_adj == reviewer_packet
    assert qa_adj == qa_packet
    assert notes == []
