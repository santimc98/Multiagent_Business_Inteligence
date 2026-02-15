from src.utils.contract_schema_registry import (
    build_contract_schema_examples_text,
    get_contract_schema_repair_action,
    apply_contract_schema_registry_repairs,
)


def test_contract_schema_registry_examples_include_core_fields() -> None:
    text = build_contract_schema_examples_text()

    assert "column_dtype_targets" in text
    assert "required_feature_selectors" in text
    assert "gate_list_example" in text


def test_contract_schema_registry_has_dtype_repair_action() -> None:
    action = get_contract_schema_repair_action("contract.column_dtype_targets")

    assert "target_dtype" in action
    assert "NOT 'type'" in action


def test_contract_schema_registry_repairs_common_shapes() -> None:
    contract = {
        "column_dtype_targets": {"age": {"type": "float64"}, "sex": "string"},
        "artifact_requirements": {
            "clean_dataset": {
                "required_feature_selectors": [
                    {"selector": "prefix:pixel_"},
                    {"pattern": "^feat_\\d+$"},
                ]
            }
        },
        "cleaning_gates": ["required_columns_present"],
    }

    repaired = apply_contract_schema_registry_repairs(contract)

    assert repaired.get("column_dtype_targets", {}).get("age", {}).get("target_dtype") == "float64"
    assert repaired.get("column_dtype_targets", {}).get("sex", {}).get("target_dtype") == "string"
    selectors = (
        repaired.get("artifact_requirements", {})
        .get("clean_dataset", {})
        .get("required_feature_selectors", [])
    )
    assert any(sel.get("type") == "prefix" for sel in selectors)
    assert any(sel.get("type") == "regex" for sel in selectors)
    gates = repaired.get("cleaning_gates", [])
    assert isinstance(gates, list) and gates
    assert gates[0].get("name") == "required_columns_present"
