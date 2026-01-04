from src.graph.graph import _normalize_alignment_check


def test_alignment_check_normalizer_builds_requirements():
    alignment_check = {
        "status": "PASS",
        "checks": [
            {"id": "req_1", "status": "PASS", "evidence": ["ok"]},
            {"name": "req_2", "status": "FAIL", "evidence": "bad"},
        ],
    }

    normalized, issues = _normalize_alignment_check(alignment_check, [])
    reqs = normalized.get("requirements")
    assert isinstance(reqs, list)

    status_map = {req.get("id"): req.get("status") for req in reqs}
    assert status_map.get("req_1") == "PASS"
    assert status_map.get("req_2") == "FAIL"
    assert "alignment_missing_requirement_status" not in issues


def test_alignment_check_normalizer_accepts_overall_status():
    alignment_check = {
        "overall_status": "PASS",
        "checks": [
            {"id": "req_alpha", "status": "PASS", "evidence": ["ok"]},
        ],
    }

    normalized, issues = _normalize_alignment_check(alignment_check, [])
    assert normalized.get("status") == "PASS"
    assert "alignment_status_invalid" not in issues
