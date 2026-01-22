import pandas as pd

from src.utils.integrity_audit import run_integrity_audit


def test_integrity_audit_validations_strings():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # V4.1: Use canonical_columns, column_roles, and validation_requirements.additional_checks
    contract = {
        "canonical_columns": ["a"],
        "column_roles": {"a": "pre_decision"},
        "validation_requirements": {
            "additional_checks": ["ranking_coherence_spearman", "numeric_range_check"]
        },
    }
    issues, stats = run_integrity_audit(df, contract)
    assert any(i["type"] == "VALIDATION_REQUIRED" for i in issues)
    # Should not raise; stats present
    assert "a" in stats
