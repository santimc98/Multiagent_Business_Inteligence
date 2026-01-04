import json
import os

from src.graph import graph as graph_mod


class _StubReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "Procedural alignment ok."}


def test_alignment_method_choice_nonblocking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.9}, f)
    alignment_payload = {
        "status": "WARN",
        "failure_mode": "method_choice",
        "summary": "Method chosen differs from expectation.",
        "requirements": [{"id": "objective_alignment", "status": "WARN", "evidence": ["method mismatch"]}],
    }
    with open(os.path.join("data", "alignment_check.json"), "w", encoding="utf-8") as f_alignment:
        json.dump(alignment_payload, f_alignment, indent=2)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "Optimize contract win-rate.",
        "execution_contract": {"alignment_requirements": [{"id": "objective_alignment", "required": True}]},
        "evaluation_spec": {"alignment_requirements": [{"id": "objective_alignment"}]},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewer())
    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    last_gate = result["last_gate_context"]
    assert "alignment_method_choice" not in last_gate.get("failed_gates", [])
    assert any("ALIGNMENT_CHECK_WARN" in entry for entry in result["feedback_history"])
