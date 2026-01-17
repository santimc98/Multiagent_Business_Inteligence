import importlib
import os

os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("MIMO_API_KEY", "test")
os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("ML_ENGINEER_PROVIDER", "openrouter")


graph = importlib.import_module("src.graph.graph")


def test_agentstate_annotations_include_routing_flags():
    annotations = graph.AgentState.__annotations__
    for key in ["ml_preflight_failed", "dataset_scale_hints", "dataset_scale"]:
        assert key in annotations


def test_check_ml_preflight_rejects_failed_flag():
    assert graph.check_ml_preflight({"ml_preflight_failed": True}) == "failed"


def test_check_ml_preflight_rejects_fallback_context():
    state = {
        "review_verdict": "REJECTED",
        "last_gate_context": {"source": "ml_preflight"},
    }
    assert graph.check_ml_preflight(state) == "failed"
