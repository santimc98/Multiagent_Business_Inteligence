import json
import os

from src.graph import graph as graph_mod


class _StubReviewerRetryNotWorth:
    """Stub that returns retry_worth_it=False, triggering downgrade to APPROVE_WITH_WARNINGS."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "LLM concern", "retry_worth_it": False}


class _StubReviewerRetryWorth:
    """Stub that returns retry_worth_it=True, keeping NEEDS_IMPROVEMENT for iteration."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "LLM concern", "retry_worth_it": True}


class _StubReviewerRetryUnknown:
    """Stub that returns retry_worth_it=None (unknown), keeping NEEDS_IMPROVEMENT for iteration."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "LLM concern"}


def test_result_evaluator_llm_nonblocking_downgrades_when_retry_not_worth(tmp_path, monkeypatch):
    """When retry_worth_it=False, NEEDS_IMPROVEMENT should downgrade to APPROVE_WITH_WARNINGS."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerRetryNotWorth())
    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert any("REVIEWER_LLM_NONBLOCKING_WARNING" in item for item in result["feedback_history"])


def test_result_evaluator_keeps_needs_improvement_when_retry_worth(tmp_path, monkeypatch):
    """When retry_worth_it=True, NEEDS_IMPROVEMENT should NOT downgrade to allow metric iteration."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerRetryWorth())
    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any("REVIEWER_LLM_NONBLOCKING_WARNING" in item for item in result["feedback_history"])


def test_result_evaluator_keeps_needs_improvement_when_retry_unknown(tmp_path, monkeypatch):
    """When retry_worth_it=None (unknown), NEEDS_IMPROVEMENT should NOT downgrade (conservative)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerRetryUnknown())
    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any("REVIEWER_LLM_NONBLOCKING_WARNING" in item for item in result["feedback_history"])
