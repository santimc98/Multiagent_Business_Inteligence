import json
from unittest.mock import MagicMock
from src.agents.execution_planner import ExecutionPlannerAgent

def test_execution_planner_artifact_schemas_scored_rows_includes_required_columns():
    planner = ExecutionPlannerAgent(api_key="mock_key")
    
    # Mock V4.1 Response simulating intelligent schema generation
    mock_response_content = {
        "contract_version": 2,
        "artifact_requirements": {
            "file_schemas": {
                "data/scored_rows.csv": {
                    "required_columns": [
                        "is_success",
                        "cluster_id",
                        "recommended_price",
                        "expected_value_at_recommendation",
                        "pred_prob_success"
                    ],
                    "allowed_name_patterns": [
                        "^recommended_.*",
                        "^expected_.*"
                    ]
                }
            }
        },
        "strategy_title": "Mock Strategy",
        "business_objective": "Mock Objective",
        "missing_columns_handling": {},
        "execution_constraints": {},
        "objective_analysis": {},
        "data_analysis": {},
        "feature_engineering_plan": {},
        "preprocessing_requirements": {},
        "validation_requirements": {},
        "leakage_execution_plan": {},
        "column_roles": {},
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {},
        "ml_engineer_runbook": {},
        "canonical_columns": [],
        "available_columns": [],
        "required_outputs": ["data/scored_rows.csv"],
        "iteration_policy": {},
        "unknowns": [],
        "assumptions": [],
        "notes_for_engineers": []
    }
    
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(mock_response_content)
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp

    strategy = {
        "required_columns": ["CurrentPhase", "1stYearAmount"],
        "analysis_type": "ranking",
        "title": "Ranked scoring for success",
    }
    
    contract = planner.generate_contract(
        strategy=strategy,
        business_objective="CurrentPhase contiene 'Contract'; optimiza el precio per segment.",
        data_summary="Column Types:\n- Categorical/Boolean: CurrentPhase\n",
        column_inventory=["CurrentPhase", "1stYearAmount", "Size", "Debtors", "Sector"],
    )
    
    # Assertions
    artifact_reqs = contract.get("artifact_requirements", {})
    artifact_schemas = artifact_reqs.get("file_schemas", {})
    assert isinstance(artifact_schemas, dict)
    
    scored_schema = artifact_schemas.get("data/scored_rows.csv")
    assert isinstance(scored_schema, dict)
    
    required_columns = scored_schema.get("required_columns") or []
    norm_required = {col.lower() for col in required_columns}
    
    assert "is_success" in norm_required
    assert "cluster_id" in norm_required
    assert any(col.lower().startswith("recommended_") for col in required_columns)
    assert "expected_value_at_recommendation" in norm_required
    assert "pred_prob_success" in norm_required
    
    patterns = scored_schema.get("allowed_name_patterns") or []
    assert any(pattern.startswith("^recommended_") for pattern in patterns)
    assert any(pattern.startswith("^expected_") for pattern in patterns)
