from src.utils.contract_accessors import get_cleaning_gates, get_qa_gates, get_reviewer_gates


def test_get_qa_gates_accepts_metric_alias_and_preserves_params():
    contract = {
        "qa_gates": [
            {"metric": "roc_auc", "severity": "hard", "threshold": 0.82},
            {"name": "stability_check", "severity": "SOFT"},
        ]
    }

    gates = get_qa_gates(contract)

    assert len(gates) == 2
    assert gates[0]["name"] == "roc_auc"
    assert gates[0]["severity"] == "HARD"
    assert gates[0]["params"].get("metric") == "roc_auc"
    assert gates[0]["params"].get("threshold") == 0.82


def test_get_reviewer_gates_accepts_check_alias():
    contract = {
        "reviewer_gates": [
            {"check": "artifact_completeness", "severity": "HARD"},
            {"rule": "contract_alignment", "severity": "SOFT"},
        ]
    }

    gates = get_reviewer_gates(contract)

    assert len(gates) == 2
    assert [g["name"] for g in gates] == ["artifact_completeness", "contract_alignment"]


def test_get_cleaning_gates_accepts_rule_alias():
    contract = {"cleaning_gates": [{"rule": "required_columns_present"}]}

    gates = get_cleaning_gates(contract)

    assert len(gates) == 1
    assert gates[0]["name"] == "required_columns_present"
    assert gates[0]["severity"] == "HARD"
