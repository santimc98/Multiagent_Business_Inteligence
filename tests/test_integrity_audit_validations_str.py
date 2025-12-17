import pandas as pd

from src.utils.integrity_audit import run_integrity_audit


def test_integrity_audit_validations_strings():
    df = pd.DataFrame({"a": [1, 2, 3]})
    contract = {"data_requirements": [{"name": "a", "role": "feature", "expected_range": [0, 10]}], "validations": ["ranking_coherence_spearman", "numeric_range_check"]}
    issues, stats = run_integrity_audit(df, contract)
    assert any(i["type"] == "VALIDATION_REQUIRED" for i in issues)
    # Should not raise; stats present
    assert "a" in stats
