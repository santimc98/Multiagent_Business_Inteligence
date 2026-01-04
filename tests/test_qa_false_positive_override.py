import types

from src.graph import graph as graph_module
from src.graph.graph import run_qa_reviewer, qa_reviewer


def test_qa_override_variance_false_positive(monkeypatch):
    code = """
import pandas as pd
df = pd.DataFrame({"y":[1,2,3]})
y = df["y"]
if y.nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
print("Mapping Summary:", {"target": "y", "features": []})
"""

    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Missing target variance guard",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": ["Add variance guard"],
        }

    def fake_static(*args, **kwargs):
        return {"status": "PASS", "facts": {}}

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    monkeypatch.setattr(graph_module, "run_static_qa_checks", fake_static)
    state = {
        "generated_code": code,
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 0,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] in ("APPROVED", "APPROVE_WITH_WARNINGS")
    assert "QA_LLM_NONBLOCKING_WARNING" in result["feedback_history"][-1]


def test_qa_fail_safe_preserves_gate_context(monkeypatch):
    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Fail",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": [],
        }

    def fake_static(*args, **kwargs):
        return {"status": "PASS", "facts": {}}

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    monkeypatch.setattr(graph_module, "run_static_qa_checks", fake_static)
    state = {
        "generated_code": "print('ok')",
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 5,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] in ("APPROVED", "APPROVE_WITH_WARNINGS")
    assert result.get("last_gate_context", {}).get("source") == "qa_reviewer"
