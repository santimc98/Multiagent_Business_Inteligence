from src.graph.graph import _resolve_allowed_patterns_for_gate


def test_allowed_patterns_handles_list_artifact_requirements():
    contract = {"artifact_requirements": []}
    assert _resolve_allowed_patterns_for_gate(contract) == []
