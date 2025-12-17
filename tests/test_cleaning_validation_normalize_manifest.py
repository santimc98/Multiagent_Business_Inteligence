from src.utils.cleaning_validation import normalize_manifest


def test_normalize_manifest_handles_list_of_dicts():
    manifest = {
        "dropped_columns": [
            {"name": "Sector", "reason": "100% null"},
            {"name": "Price", "reason": "constant"},
        ],
        "conversions": {"sector": "numeric_currency"},
        "column_mapping": {"Sector": "sector"},
    }

    norm = normalize_manifest(manifest)
    assert norm["dropped_columns"]["sector"]["reason"] == "100% null"
    assert norm["conversions"]["sector"] == "numeric_currency"
    assert norm["column_mapping"]["Sector"] == "sector"
