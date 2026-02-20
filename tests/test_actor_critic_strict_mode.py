import json

from src.graph import graph as graph_mod


class _StubReviewerApproved:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "ok"}


class _StubQAApproved:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "QA passed",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


def test_check_evaluation_ignores_results_advisor_stop_in_strict_mode(monkeypatch) -> None:
    state = {
        "review_verdict": "APPROVED",
        "execution_output": "OK",
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": True,
        "results_last_result": {"iteration_recommendation": {"action": "STOP"}},
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.82,
            "baseline_value": 0.82,
        },
    }
    route = graph_mod.check_evaluation(state)
    assert route == "approved"
    assert state.get("stop_reason") != "RESULTS_ADVISOR_STOP"


def test_retry_handler_skips_legacy_prepare_improvement_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setenv("ACTOR_CRITIC_IMPROVEMENT_STRICT", "1")
    state = {
        "last_iteration_type": "metric",
        "improvement_attempt_count": 1,
        "ml_improvement_round_active": False,
        "feedback_history": [],
    }
    result = graph_mod.retry_handler(state)
    assert isinstance(result, dict)
    assert result.get("strategy_reselected") is False


def test_run_result_evaluator_clears_iteration_recommendation_in_strict_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "metrics.json").write_text(json.dumps({"metric": 0.9}), encoding="utf-8")

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_insights",
        lambda _ctx: {
            "summary_lines": ["ok"],
            "risks": [],
            "recommendations": [],
            "iteration_recommendation": {"action": "STOP", "mode": "improve"},
        },
    )

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }
    result = graph_mod.run_result_evaluator(state)
    results_last = result.get("results_last_result", {})
    assert results_last.get("iteration_recommendation") == {}
    review_stack = result.get("ml_review_stack", {})
    assert review_stack.get("results_advisor", {}).get("iteration_recommendation") == {}
