from src.utils.contract_accessors import get_outcome_columns


def test_get_outcome_columns_reads_validation_requirements_target_column():
    contract = {
        "validation_requirements": {
            "target_column": "target",
        }
    }
    assert get_outcome_columns(contract) == ["target"]


def test_get_outcome_columns_prefers_explicit_outcome_columns():
    contract = {
        "outcome_columns": ["claim_flag"],
        "validation_requirements": {
            "target_column": "target",
        },
    }
    assert get_outcome_columns(contract) == ["claim_flag"]
