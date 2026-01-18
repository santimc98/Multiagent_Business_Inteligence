from src.utils.contract_validator import normalize_artifact_requirements


def test_conceptual_required_outputs_move_to_reporting_requirements():
    contract = {"required_outputs": ["Local Explanation (SHAP/LIME)"]}
    artifact_requirements, warnings = normalize_artifact_requirements(contract)
    required_files = artifact_requirements.get("required_files", [])
    assert not any(
        str(item.get("path", "")) == "Local Explanation (SHAP/LIME)" for item in required_files
    )
    reporting = contract.get("reporting_requirements", {})
    conceptual = reporting.get("conceptual_outputs", [])
    assert "Local Explanation (SHAP/LIME)" in conceptual
    assert any(w.get("action") == "moved_to_conceptual_outputs" for w in warnings)
