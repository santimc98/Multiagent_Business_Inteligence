from src.agents.execution_planner import _apply_deterministic_repairs


def test_deterministic_repair_column_dtype_targets_aliases() -> None:
    contract = {
        "column_dtype_targets": {
            "age": {"type": "float64", "nullable": False},
            "sex": "string",
            "score": {"dtype": "float64"},
        }
    }

    repaired = _apply_deterministic_repairs(contract)
    targets = repaired.get("column_dtype_targets", {})

    assert targets.get("age", {}).get("target_dtype") == "float64"
    assert targets.get("sex", {}).get("target_dtype") == "string"
    assert targets.get("score", {}).get("target_dtype") == "float64"
    assert "type" not in targets.get("age", {})
    assert "dtype" not in targets.get("score", {})


def test_deterministic_repair_selectors_infers_missing_type() -> None:
    contract = {
        "artifact_requirements": {
            "clean_dataset": {
                "required_feature_selectors": [
                    {"prefix": "feature_"},
                    {"selector": "regex:^pixel_\\d+$"},
                    {"type": "list", "columns": "feature_a"},
                ]
            }
        }
    }

    repaired = _apply_deterministic_repairs(contract)
    selectors = (
        repaired.get("artifact_requirements", {})
        .get("clean_dataset", {})
        .get("required_feature_selectors", [])
    )

    assert isinstance(selectors, list)
    assert any(sel.get("type") == "prefix" and sel.get("value") == "feature_" for sel in selectors)
    assert any(sel.get("type") == "regex" and sel.get("pattern") == "^pixel_\\d+$" for sel in selectors)
    assert any(sel.get("type") == "list" and sel.get("columns") == ["feature_a"] for sel in selectors)


def test_deterministic_repair_gates_normalizes_shape() -> None:
    contract = {
        "qa_gates": [
            "auc_min_gate",
            {"metric": "accuracy", "severity": "hard", "threshold": 0.8},
        ]
    }

    repaired = _apply_deterministic_repairs(contract)
    gates = repaired.get("qa_gates", [])

    assert isinstance(gates, list)
    assert gates[0]["name"] == "auc_min_gate"
    assert gates[0]["severity"] == "HARD"
    assert isinstance(gates[0]["params"], dict)
    assert gates[1]["name"] == "accuracy"
    assert gates[1]["severity"] == "HARD"
    assert gates[1]["params"].get("threshold") == 0.8


def test_deterministic_repair_derived_columns_accepts_object_mapping() -> None:
    contract = {
        "derived_columns": {
            "is_success": {"source_column": "status"},
            "amount_x_discount": {"source": "amount,discount"},
        }
    }

    repaired = _apply_deterministic_repairs(contract)

    assert repaired.get("derived_columns") == ["is_success", "amount_x_discount"]


def test_deterministic_repair_scope_promotes_full_pipeline_when_both_segments_present() -> None:
    contract = {
        "scope": "cleaning_only",
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
        "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
        "qa_gates": [{"name": "no_nulls_target", "severity": "HARD", "params": {"column": "target"}}],
        "validation_requirements": {"method": "cross_validation"},
        "ml_engineer_runbook": "Train a model and produce metrics.",
    }

    repaired = _apply_deterministic_repairs(contract)

    assert repaired.get("scope") == "full_pipeline"
