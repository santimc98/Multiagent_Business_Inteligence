from copy import deepcopy

from src.utils.contract_validator import validate_contract_readonly


def _base_contract():
    return {
        "contract_version": "4.1",
        "strategy_title": "Risk Scoring",
        "business_objective": "Build a risk score and decision policy",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": ["target"],
            "rationale": "baseline",
        },
        "artifact_requirements": {
            "required_files": [
                {"path": "data/metrics.json"},
                {"path": "data/scored_rows.csv"},
            ]
        },
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
        "validation_requirements": {
            "primary_metric": "accuracy",
            "metrics_to_report": ["accuracy"],
        },
        "qa_gates": [{"name": "benchmark_metric", "severity": "HARD"}],
        "cleaning_gates": [{"name": "schema_integrity", "severity": "HARD"}],
        "reviewer_gates": [{"name": "artifact_completeness", "severity": "HARD"}],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "ml_engineer_runbook": {"steps": ["train", "evaluate", "persist"]},
        "iteration_policy": {"max_iterations": 2},
    }


def test_validate_contract_readonly_accepts_valid_role_bucket_contract():
    result = validate_contract_readonly(_base_contract())
    assert result.get("accepted") is True
    assert result.get("status") in {"ok", "warning"}


def test_validate_contract_readonly_rejects_inverted_column_roles():
    contract = _base_contract()
    contract["column_roles"] = {"feature_a": "pre_decision", "target": "outcome"}

    result = validate_contract_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.column_roles_format" in rules or "contract.role_ontology" in rules


def test_validate_contract_readonly_rejects_non_path_required_outputs():
    contract = _base_contract()
    contract["required_outputs"] = ["predicted_class", "confidence_score"]

    result = validate_contract_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.required_outputs_path" in rules


def test_validate_contract_readonly_accepts_required_outputs_object_entries():
    contract = _base_contract()
    contract["required_outputs"] = [
        {
            "path": "data/metrics.json",
            "required": True,
            "owner": "ml_engineer",
            "kind": "metrics",
        },
        {
            "path": "data/scored_rows.csv",
            "required": True,
            "owner": "ml_engineer",
            "kind": "predictions",
        },
    ]

    result = validate_contract_readonly(contract)

    assert result.get("accepted") is True
    assert result.get("status") in {"ok", "warning"}


def test_validate_contract_readonly_rejects_required_outputs_object_without_path():
    contract = _base_contract()
    contract["required_outputs"] = [{"output": "data/metrics.json"}]

    result = validate_contract_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.required_outputs_path" in rules


def test_validate_contract_readonly_rejects_alias_role_key():
    contract = deepcopy(_base_contract())
    contract["column_roles"] = {
        "pre_decision": ["feature_a"],
        "decision": [],
        "outcome": ["target"],
        "audit_only": [],
    }

    result = validate_contract_readonly(contract)

    assert result.get("accepted") is False
    assert any(
        str(issue.get("rule")) == "contract.role_ontology"
        for issue in result.get("issues", [])
        if isinstance(issue, dict)
    )
