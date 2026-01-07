import pytest
from unittest.mock import MagicMock
import json
from src.agents.execution_planner import ExecutionPlannerAgent

def test_execution_planner_target_labels_from_objective_es():
    # Use mock to simulate LLM reasoning extracted from Spanish objective
    planner = ExecutionPlannerAgent(api_key="mock_key")
    
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "contract_version": 2,
        "feature_engineering_plan": {
            "derived_columns": [
                {
                    "name": "CurrentPhase_target",
                    "derivation_type": "rule_from_outcome",
                    "source_column": "CurrentPhase",
                    "positive_values": ["Contract"]
                }
            ]
        },
        "column_roles": {
            "outcome": ["CurrentPhase"]
        },
        "data_analysis": {},
        "missing_columns_handling": {},
        "execution_constraints": {},
        "objective_analysis": {},
        "preprocessing_requirements": {},
        "validation_requirements": {},
        "leakage_execution_plan": {},
        "optimization_specification": None,
        "segmentation_constraints": None,
        "data_limited_mode": {},
        "allowed_feature_sets": {},
        "artifact_requirements": {},
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {},
        "ml_engineer_runbook": {},
        "available_columns": [], 
        "canonical_columns": [],
        "required_outputs": [],
        "iteration_policy": {},
        "unknowns": [],
        "assumptions": [],
        "notes_for_engineers": []
    })
    
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp
    
    strategy = {"analysis_type": "predictive", "required_columns": ["CurrentPhase", "Size"]}
    contract = planner.generate_contract(
        strategy=strategy,
        business_objective="El campo “CurrentPhase” señala la fase final...",
        column_inventory=["CurrentPhase", "Size"]
    )
    
    # Assert V4.1 Structure
    fep = contract.get("feature_engineering_plan", {})
    derived = fep.get("derived_columns", [])
    target = next((d for d in derived if isinstance(d, dict) and d.get("positive_values")), None)
    
    assert target is not None
    assert "Contract" in target.get("positive_values")
