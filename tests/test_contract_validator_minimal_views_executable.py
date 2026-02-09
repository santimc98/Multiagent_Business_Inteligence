from src.utils.contract_validator import validate_contract_minimal_readonly


def _base_full_pipeline_contract():
    return {
        "scope": "full_pipeline",
        "strategy_title": "Risk Scoring",
        "business_objective": "Predict risk and support operational decisions.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
        },
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["id", "feature_a", "target"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            },
            "required_files": [
                {"path": "data/cleaned_data.csv"},
                {"path": "data/cleaning_manifest.json"},
                {"path": "reports/performance_metrics.json"},
                {"path": "outputs/risk_scores_and_decisions.csv"},
            ],
        },
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/cleaning_manifest.json",
            "reports/performance_metrics.json",
            "outputs/risk_scores_and_decisions.csv",
        ],
        "cleaning_gates": ["schema_integrity"],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "qa_gates": ["metric_threshold"],
        "reviewer_gates": ["artifact_completeness"],
        "validation_requirements": {"primary_metric": "normalized_gini", "method": "stratified_kfold"},
        "evaluation_spec": {"objective_type": "classification", "primary_metric": "normalized_gini"},
        "ml_engineer_runbook": {"steps": ["train", "evaluate", "persist"]},
        "objective_analysis": {"problem_type": "prediction"},
        "iteration_policy": {"max_iterations": 2},
    }


def test_validate_contract_minimal_readonly_rejects_missing_de_manifest_path():
    contract = _base_full_pipeline_contract()
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset.pop("output_manifest_path", None)
    contract["required_outputs"] = [
        path for path in contract.get("required_outputs", []) if "manifest" not in str(path).lower()
    ]
    contract["artifact_requirements"]["required_files"] = [
        item
        for item in contract["artifact_requirements"].get("required_files", [])
        if "manifest" not in str(item.get("path", "")).lower()
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.de_view_manifest_path" in rules


def test_validate_contract_minimal_readonly_rejects_unknown_ml_objective():
    contract = _base_full_pipeline_contract()
    contract.pop("objective_analysis", None)
    contract.pop("evaluation_spec", None)
    contract["required_outputs"] = [
        "data/cleaned_data.csv",
        "data/cleaning_manifest.json",
        "reports/performance_metrics.json",
        "artifacts/model_bundle.bin",
    ]
    contract["artifact_requirements"]["required_files"] = [
        {"path": path} for path in contract["required_outputs"]
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.ml_view_objective_type" in rules


def test_validate_contract_minimal_readonly_accepts_executable_views_contract():
    contract = _base_full_pipeline_contract()

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    assert str(result.get("status")).lower() in {"ok", "warning"}


def test_validate_contract_minimal_readonly_allows_missing_iteration_policy_with_warning():
    contract = _base_full_pipeline_contract()
    contract.pop("iteration_policy", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy" in rules


def test_validate_contract_minimal_readonly_allows_missing_evaluation_spec_for_ml_scope_with_warning():
    contract = _base_full_pipeline_contract()
    contract.pop("evaluation_spec", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.evaluation_spec" in rules


def test_validate_contract_minimal_readonly_accepts_iteration_policy_alias_keys():
    contract = _base_full_pipeline_contract()
    contract["iteration_policy"] = {
        "max_pipeline_iterations": 3,
        "gate_retry_limit": 2,
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy_limits" not in rules


def test_validate_contract_minimal_readonly_accepts_max_retries_alias():
    contract = _base_full_pipeline_contract()
    contract["iteration_policy"] = {
        "max_retries": 4,
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy_limits" not in rules
