from copy import deepcopy

from src.agents.execution_planner import _validate_repair_revalidate_loop
from src.utils.contract_validator import validate_contract_minimal_readonly


def _base_contract_with_typical_errors() -> dict:
    return {
        "scope": "full_pipeline",
        "strategy_title": "Risk Scoring",
        "business_objective": "Predict risk and support decisions.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature_a", "target"],
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
            "clean_dataset": {
                "required_columns": ["id", "feature_a", "target"],
                "required_feature_selectors": ["prefix:feature_"],  # malformed on purpose
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
        "required_outputs": ["Priority ranking"],  # invalid path on purpose
        "required_output_artifacts": [
            {"path": "reports/performance_metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
            {"path": "outputs/risk_scores_and_decisions.csv", "required": True, "owner": "ml_engineer", "kind": "predictions"},
        ],
        "spec_extraction": {
            "deliverables": [
                {"path": "reports/performance_metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
                {"path": "outputs/risk_scores_and_decisions.csv", "required": True, "owner": "ml_engineer", "kind": "predictions"},
            ]
        },
        "column_dtype_targets": {
            "id": {"type": "string"},  # wrong key on purpose
            "feature_a": {"type": "float64"},  # wrong key on purpose
            "target": {"type": "float64", "nullable": True},  # wrong key on purpose
        },
        "cleaning_gates": [{"name": "schema_integrity", "severity": "HARD"}],
        "qa_gates": [{"name": "benchmark_metric", "severity": "HARD"}],
        "reviewer_gates": [{"name": "artifact_completeness", "severity": "HARD"}],
        "validation_requirements": {"primary_metric": "accuracy"},
        "evaluation_spec": {"objective_type": "classification"},
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "ml_engineer_runbook": {"steps": ["train", "evaluate", "persist"]},
        "objective_analysis": {"problem_type": "prediction"},
        "iteration_policy": {"max_iterations": 2},
    }


def test_validate_repair_revalidate_loop_repairs_typical_contract_errors():
    contract = _base_contract_with_typical_errors()
    original_deliverables = deepcopy(contract["spec_extraction"]["deliverables"])
    original_rich_outputs = deepcopy(contract["required_output_artifacts"])

    def _repair_provider(current_contract: dict, validation_result: dict, hints: list[str], attempt: int):
        assert attempt == 1
        assert hints  # must include compact actionable hints
        return {
            "changes": {
                "required_outputs": [
                    "data/cleaned_data.csv",
                    "data/cleaning_manifest.json",
                    "reports/performance_metrics.json",
                    "outputs/risk_scores_and_decisions.csv",
                ],
                "column_dtype_targets": {
                    "id": {"target_dtype": "string"},
                    "feature_a": {"target_dtype": "float64"},
                    "target": {"target_dtype": "float64", "nullable": True},
                },
                "artifact_requirements": {
                    "clean_dataset": {
                        "required_feature_selectors": [{"type": "prefix", "value": "feature_"}],
                    }
                },
            }
        }

    repaired, validation, trace = _validate_repair_revalidate_loop(
        contract=contract,
        validator_fn=lambda payload: validate_contract_minimal_readonly(payload),
        repair_provider=_repair_provider,
        max_iterations=2,
    )

    assert validation.get("accepted") is True
    assert isinstance(repaired.get("required_outputs"), list)
    assert all(isinstance(path, str) for path in repaired.get("required_outputs", []))
    assert repaired.get("spec_extraction", {}).get("deliverables") == original_deliverables
    assert repaired.get("required_output_artifacts") == original_rich_outputs
    assert len(trace) >= 2
