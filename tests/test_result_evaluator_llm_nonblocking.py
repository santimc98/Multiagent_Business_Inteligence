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

class _StubQAApproved:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "QA Passed",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }

class _StubQARejectedNoGates:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "REJECTED",
            "feedback": "Output schema mismatch",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


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


def test_result_evaluator_hard_gate_column_presence_blocks_success(tmp_path, monkeypatch):
    """
    HARD QA gate: explanation column missing in scored_rows.csv -> must not approve.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.7}, f)
    with open(os.path.join("data", "scored_rows.csv"), "w", encoding="utf-8") as f:
        f.write("top_drivers\nfoo\n")

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {
            "qa_gates": [
                {
                    "name": "explanation_column_presence",
                    "severity": "HARD",
                    "params": {"target_file": "data/scored_rows.csv", "column": "explanation"},
                }
            ]
        },
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any("QA_GATE_FAIL" in item for item in result["feedback_history"])

    state.update(result)
    assert graph_mod.check_evaluation(state) == "retry"


def test_result_evaluator_qa_rejected_blocks_metric_downgrade(tmp_path, monkeypatch):
    """
    QA REJECTED (even without failed_gates) must block metric-only downgrade.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('hello')",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQARejectedNoGates())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any("CODE_AUDIT_REJECTED" in item for item in result["feedback_history"])
    assert "hard_failures" in result and "qa_rejected" in result["hard_failures"]

    state.update(result)
    assert graph_mod.check_evaluation(state) == "retry"
