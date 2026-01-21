from src.utils.column_sets import build_column_sets, expand_column_sets


def test_column_sets_prefix_range():
    columns = [f"feat{i}" for i in range(100)] + ["id", "target"]
    roles = {"id": "id_like", "target": "target_candidate"}

    sets_spec = build_column_sets(columns, roles=roles)
    assert sets_spec["sets"], "Expected at least one column set"
    selector = sets_spec["sets"][0]["selector"]
    assert selector["type"] == "prefix_numeric_range"
    assert selector["prefix"] == "feat"
    assert selector["start"] == 0
    assert selector["end"] == 99

    expanded = expand_column_sets(columns, sets_spec)
    expanded_cols = expanded["expanded_feature_columns"]
    assert len(expanded_cols) == 100
    assert "id" not in expanded_cols
    assert "target" not in expanded_cols
