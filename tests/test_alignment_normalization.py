from src.graph.graph import _normalize_alignment_check


def test_alignment_check_accepts_string_requirements():
    alignment_check = {"status": "PASS", "requirements": []}
    normalized, issues = _normalize_alignment_check(alignment_check, ["objective_alignment"])
    assert isinstance(normalized, dict)
    reqs = normalized.get("requirements", [])
    assert isinstance(reqs, list)
    assert any(r.get("id") == "objective_alignment" for r in reqs)
