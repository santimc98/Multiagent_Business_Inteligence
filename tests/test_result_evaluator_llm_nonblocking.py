"""
Tests for result evaluator behavior with metric iteration DISABLED.

With metric iteration disabled:
- NEEDS_IMPROVEMENT for metric-only issues -> APPROVE_WITH_WARNINGS (no retry)
- NEEDS_IMPROVEMENT for compliance issues (audit rejected, missing outputs) -> stays NEEDS_IMPROVEMENT (allows retry)
"""
import json
import os

from src.graph import graph as graph_mod


class _StubReviewerMetricIssue:
    """Stub that returns NEEDS_IMPROVEMENT for metric-only issue (no audit rejection)."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "Metrics below threshold", "retry_worth_it": True}


class _StubReviewerApproved:
    """Stub that returns APPROVED."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "All good"}


def test_result_evaluator_metric_issue_downgrades_to_approve_with_warnings(tmp_path, monkeypatch):
    """
    When NEEDS_IMPROVEMENT is for metric-only issues (iteration_type='metric'),
    it should downgrade to APPROVE_WITH_WARNINGS since metric iteration is disabled.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",  # No traceback -> not compliance
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    result = graph_mod.run_result_evaluator(state)

    # With metric iteration disabled, metric-only issues get downgraded
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    # Should have the metric iteration disabled message
    assert any("Metric iteration disabled" in item for item in result["feedback_history"])


def test_result_evaluator_compliance_issue_keeps_needs_improvement(tmp_path, monkeypatch):
    """
    When NEEDS_IMPROVEMENT is for compliance issues (traceback, audit rejection),
    it should stay NEEDS_IMPROVEMENT to allow retry for compliance fixes.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "Traceback (most recent call last):\n  File...\nValueError: something wrong",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    result = graph_mod.run_result_evaluator(state)

    # Compliance issues (traceback) should keep NEEDS_IMPROVEMENT for retry
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    # Should NOT have metric iteration disabled message (this is compliance)
    assert not any("Metric iteration disabled" in item for item in result["feedback_history"])


def test_result_evaluator_approved_stays_approved(tmp_path, monkeypatch):
    """When APPROVED, it should stay APPROVED (no changes)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.9}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "APPROVED"
