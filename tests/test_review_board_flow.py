import json
import os

from src.graph import graph as graph_mod


class _StubBoardWarnings:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, _context):
        return {
            "status": "APPROVE_WITH_WARNINGS",
            "summary": "Runtime failed but report can proceed with caveats.",
            "failed_areas": ["runtime"],
            "required_actions": ["Document runtime failure in report."],
            "confidence": "high",
            "evidence": [],
        }


class _RuntimePathReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        raise AssertionError("evaluate_results must be skipped when runtime markers are present.")


def test_check_evaluation_terminal_runtime_stops():
    state = {
        "runtime_fix_terminal": True,
        "review_verdict": "NEEDS_IMPROVEMENT",
    }
    assert graph_mod.check_evaluation(state) == "approved"


def test_run_review_board_persists_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardWarnings())
    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "review_feedback": "Runtime failure detected.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": ["runtime_failure"], "required_fixes": []},
        "ml_review_stack": {
            "runtime": {"status": "FAILED_RUNTIME", "runtime_fix_terminal": True},
            "result_evaluator": {"status": "NEEDS_IMPROVEMENT"},
            "reviewer": {"status": "SKIPPED"},
            "qa_reviewer": {"status": "SKIPPED"},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert os.path.exists("data/review_board_verdict.json")
    with open("data/review_board_verdict.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["status"] == "APPROVE_WITH_WARNINGS"
    assert payload["final_review_verdict"] == "APPROVE_WITH_WARNINGS"


def test_run_result_evaluator_runtime_failure_builds_review_stack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "reviewer", _RuntimePathReviewer())
    state = {
        "execution_output": "HEAVY_RUNNER_ERROR: timeout while streaming artifacts",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert "ml_review_stack" in result
    assert result["ml_review_stack"]["runtime"]["status"] == "FAILED_RUNTIME"
    assert result["ml_review_stack"]["result_evaluator"]["status"] == result["ml_review_stack"]["final_pre_board"]["status"]
    assert result["ml_review_stack"]["result_evaluator"]["raw_status"] == "NEEDS_IMPROVEMENT"
    assert isinstance(result["ml_review_stack"].get("deterministic_facts"), dict)
    assert os.path.exists("data/ml_review_stack.json")
