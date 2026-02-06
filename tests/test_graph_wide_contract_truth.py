from src.graph.graph import (
    _apply_wide_dataset_contract_truth,
    _resolve_explicit_columns_for_wide_dataset,
)


def test_apply_wide_dataset_truth_preserves_canonical_columns():
    state = {}
    contract = {"canonical_columns": ["label", "pixel0", "pixel1"]}
    contract_min = {"canonical_columns": ["label", "pixel0", "pixel1"]}

    _apply_wide_dataset_contract_truth(
        state=state,
        contract=contract,
        contract_min=contract_min,
        n_columns=786,
    )

    assert contract_min["canonical_columns"] == ["label", "pixel0", "pixel1"]
    assert contract_min["available_columns"] == ["label", "pixel0", "pixel1"]
    assert contract_min["dataset_truth_ref"]["n_columns"] == 786


def test_apply_wide_dataset_truth_uses_explicit_columns_without_overwriting_canonical():
    state = {"column_sets": {"explicit_columns": ["label"]}}
    contract = {"canonical_columns": ["label", "pixel0", "pixel1"]}
    contract_min = {"canonical_columns": ["label", "pixel0", "pixel1"]}

    _apply_wide_dataset_contract_truth(
        state=state,
        contract=contract,
        contract_min=contract_min,
        n_columns=786,
    )

    assert contract_min["canonical_columns"] == ["label", "pixel0", "pixel1"]
    assert contract_min["available_columns"] == ["label"]


def test_resolve_explicit_columns_loads_from_artifact_when_state_missing(monkeypatch):
    state = {}

    def _fake_load(path):
        if path == "data/column_sets.json":
            return {"explicit_columns": ["target_col"]}
        return {}

    monkeypatch.setattr("src.graph.graph._load_json_safe", _fake_load)

    explicit = _resolve_explicit_columns_for_wide_dataset(state)

    assert explicit == ["target_col"]
    assert state["column_sets"]["explicit_columns"] == ["target_col"]
