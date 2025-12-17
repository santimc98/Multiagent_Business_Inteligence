import pandas as pd
from src.utils.cleaning_validation import detect_destructive_drop


def test_detects_destructive_drop_on_text_column(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "sector,other\nInformática, Electrónica y Óptica,1\nAutomoción y Moda,2\nEnergía y Gas,3\n",
        encoding="utf-8"
    )

    manifest = {
        "conversions": {"sector": "numeric_currency"},
        "dropped_columns": [{"name": "sector", "reason": "100% null"}],
        "column_mapping": {"sector": "sector"},
        "original_columns": ["sector", "other"],
    }

    issues = detect_destructive_drop(
        manifest,
        ["sector"],
        str(raw_path),
        {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    )

    assert issues, "Expected destructive conversion issue to be detected"
    assert issues[0]["column"] == "sector"
