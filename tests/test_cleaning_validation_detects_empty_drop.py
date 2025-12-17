import pandas as pd

from src.utils.cleaning_validation import detect_destructive_drop


def test_detects_empty_drop_on_nonempty_required_col(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "visits,other\n10,1\n20,2\n30,3\n",
        encoding="utf-8"
    )

    manifest = {
        "conversions": {"visits": "to_numeric"},
        "dropped_columns": [{"name": "visits", "reason": "empty"}],
        "column_mapping": {"visits": "visits"},
        "original_columns": ["visits", "other"],
    }

    issues = detect_destructive_drop(
        manifest,
        ["visits"],
        str(raw_path),
        {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    )

    assert issues, "Expected destructive empty drop to be detected"
    assert issues[0]["raw_non_null_frac"] >= 1.0
