import pytest
from unittest.mock import MagicMock
import json
from src.agents.execution_planner import ExecutionPlannerAgent

def test_execution_planner_derives_target_from_status_column():
    planner = ExecutionPlannerAgent(api_key="mock_key")
    
    # Mock LLM response
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "contract_version": 2,
        "feature_engineering_plan": {
            "derived_columns": [
                {
                    "name": "is_success",
                    "source_column": "CurrentPhase",
                    "derivation_type": "rule_from_outcome"
                }
            ]
        },
        "column_roles": { "outcome": ["CurrentPhase"] },
        "data_analysis": {}, "missing_columns_handling": {}, "execution_constraints": {}, "objective_analysis": {}, "preprocessing_requirements": {}, "validation_requirements": {}, "leakage_execution_plan": {}, "optimization_specification": None, "segmentation_constraints": None, "data_limited_mode": {}, "allowed_feature_sets": {}, "artifact_requirements": {}, "qa_gates": [], "reviewer_gates": [], "data_engineer_runbook": {}, "ml_engineer_runbook": {}, "available_columns": [], "canonical_columns": [], "required_outputs": [], "iteration_policy": {}, "unknowns": [], "assumptions": [], "notes_for_engineers": []
    })
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp

    strategy = {"required_columns": ["CurrentPhase"], "analysis_type": "predictive", "title": "Conversion"}
    contract = planner.generate_contract(strategy=strategy, data_summary="", business_objective="", column_inventory=[])

    # Assert V4.1
    fep = contract.get("feature_engineering_plan", {})
    derived = fep.get("derived_columns", [])
    target = next((d for d in derived if d.get("name") == "is_success"), None)
    assert target is not None
    assert target.get("source_column") == "CurrentPhase"


def test_execution_planner_derives_positive_labels_from_objective_contains():
    planner = ExecutionPlannerAgent(api_key="mock_key")
    
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "contract_version": 2,
        "feature_engineering_plan": {
            "derived_columns": [
                {
                    "name": "is_success",
                    "source_column": "CurrentPhase",
                    "positive_values": ["Contract"]
                }
            ]
        },
        "column_roles": { "outcome": ["CurrentPhase"] },
        "data_analysis": {}, "missing_columns_handling": {}, "execution_constraints": {}, "objective_analysis": {}, "preprocessing_requirements": {}, "validation_requirements": {}, "leakage_execution_plan": {}, "optimization_specification": None, "segmentation_constraints": None, "data_limited_mode": {}, "allowed_feature_sets": {}, "artifact_requirements": {}, "qa_gates": [], "reviewer_gates": [], "data_engineer_runbook": {}, "ml_engineer_runbook": {}, "available_columns": [], "canonical_columns": [], "required_outputs": [], "iteration_policy": {}, "unknowns": [], "assumptions": [], "notes_for_engineers": []
    })
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp
    
    contract = planner.generate_contract(strategy={}, data_summary="", business_objective="", column_inventory=[])
    
    fep = contract.get("feature_engineering_plan", {})
    target = next((d for d in fep.get("derived_columns", []) if d.get("positive_values")), None)
    assert target is not None
    assert "Contract" in target.get("positive_values")
