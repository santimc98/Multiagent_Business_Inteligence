from src.utils.context_pack import build_context_pack


def test_context_pack_long_list_compression():
    columns = [f"col_{i}" for i in range(800)]
    state = {
        "run_id": "test123",
        "csv_sep": ",",
        "csv_decimal": ".",
        "csv_encoding": "utf-8",
        "column_inventory": columns,
        "execution_contract": {
            "required_outputs": columns,
            "decisioning_requirements": {
                "output": {"required_columns": columns},
            },
        },
    }

    pack = build_context_pack("test", state)
    assert "data/column_sets.json" in pack
    assert "data/column_inventory.json" in pack
    assert pack.count("col_") <= 40
