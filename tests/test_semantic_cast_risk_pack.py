from src.utils.semantic_cast_risk import (
    build_column_semantic_cast_risk_pack,
    summarize_column_semantic_cast_risk_pack,
)
from src.agents.execution_planner import (
    _column_semantic_cast_validation_result,
    _validate_column_semantic_cast_coherence,
)


def test_semantic_cast_risk_pack_flags_boolean_like_numeric_target():
    pack = build_column_semantic_cast_risk_pack(
        {
            "column_facts": [
                {
                    "name": "Activo",
                    "type_hint": "categorical",
                    "top_values": [{"value": "TRUE"}, {"value": "FALSE"}],
                    "top_value_share": 0.99,
                    "missing_frac": 0.0,
                }
            ]
        },
        column_dtype_targets={"Activo": {"target_dtype": "float64"}},
    )

    assert pack["risk_count"] == 1
    risk = pack["risks"][0]
    assert risk["column"] == "Activo"
    assert "boolean_like_values" in risk["risk_kinds"]
    assert "categorical_to_numeric_target" in risk["risk_kinds"]
    assert risk["target_dtype"] == "numeric"

    summary = summarize_column_semantic_cast_risk_pack(pack)
    assert "Activo" in summary
    assert "advisory_context_only" in summary


def test_planner_semantic_cast_validator_warns_without_blocking():
    contract = {"shared": {"column_dtype_targets": {"Activo": "float64"}}}
    data_profile = {
        "column_semantic_cast_risk_pack": {
            "risks": [
                {
                    "column": "Activo",
                    "risk_kinds": ["boolean_like_values"],
                    "target_dtype": "numeric",
                }
            ]
        }
    }

    validation = _validate_column_semantic_cast_coherence(contract, data_profile)
    result = _column_semantic_cast_validation_result(validation)

    assert validation["status"] == "violations"
    assert result["status"] == "warning"
    assert result["accepted"] is True
    assert result["issues"][0]["rule"].startswith("contract.column_semantic_cast.")
