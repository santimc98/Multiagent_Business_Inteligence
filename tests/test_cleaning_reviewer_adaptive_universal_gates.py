import pandas as pd

from src.agents.cleaning_reviewer import _evaluate_gates_deterministic, normalize_gate_name


def test_id_integrity_excludes_target_columns_from_identifier_check():
    df = pd.DataFrame(
        {
            "id": ["0001", "0002", "0003", "0004"],
            "identity_hate": [0.0, 1.0, 0.0, 0.0],
            "__split": ["train", "train", "train", "train"],
        }
    )
    gates = [
        {
            "name": "id_integrity",
            "severity": "HARD",
            "params": {"detect_scientific_notation": True, "min_samples": 1},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest={},
        raw_sample=None,
        column_roles={
            "id": ["id"],
            "targets": ["identity_hate"],
            "split_indicator": ["__split"],
        },
        allowed_feature_sets={},
    )

    assert result["status"] == "APPROVED"
    assert "id_integrity" not in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "id_integrity"
    )
    evidence = gate_entry.get("evidence") or {}
    assert "id" in (evidence.get("candidate_columns") or [])
    assert "identity_hate" not in (evidence.get("candidate_columns") or [])


def test_row_count_sanity_skips_when_drop_matches_label_null_listwise_pattern():
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "target_a": [0.0, 1.0],
            "target_b": [1.0, 0.0],
            "__split": ["train", "train"],
        }
    )
    gates = [
        {
            "name": "row_count_sanity",
            "severity": "SOFT",
            "params": {"max_drop_pct": 5.0},
        }
    ]
    manifest = {
        "row_counts": {
            "initial": 1000,
            "final": 500,
            "dropped": 500,
            "dropped_reason": "null_label_removal (listwise deletion on toxicity labels)",
        },
        "gate_results": {
            "null_label_removal": {
                "status": "PASSED",
                "rows_removed": 500,
            }
        },
        "null_stats": {
            "before": {"id": 0, "target_a": 500, "target_b": 500, "__split": 0},
            "after": {"id": 0, "target_a": 0, "target_b": 0, "__split": 0},
        },
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={
            "id": ["id"],
            "targets": ["target_a", "target_b"],
            "split_indicator": ["__split"],
        },
        allowed_feature_sets={},
    )

    assert result["status"] == "APPROVED"
    assert "row_count_sanity" not in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "row_count_sanity"
    )
    evidence = gate_entry.get("evidence") or {}
    assert evidence.get("applies_if") is False
    assert evidence.get("skip_reason") == "drop_explained_by_label_null_listwise_removal"
