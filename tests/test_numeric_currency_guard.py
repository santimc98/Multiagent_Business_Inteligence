import pandas as pd

from src.utils.type_inference import profile_numeric_currency_candidate, safe_convert_numeric_currency


def test_textual_with_commas_not_currency():
    series = pd.Series(
        [
            "Informática, Electrónica y Óptica",
            "Servicios, Energía y Gas",
            "Automoción, Diseño y Moda",
        ]
    )
    profile = profile_numeric_currency_candidate(series)
    assert profile["is_numeric_currency"] is False
    assert profile["parse_success_rate"] == 0 or profile["digits_ratio"] < 0.6


def test_revert_on_low_parse_success():
    series = pd.Series(["123", "abc", "xyz"])
    converted, info = safe_convert_numeric_currency(series, parse_success_threshold=0.9, post_drop_threshold=0.9)
    assert info["reverted"] is True
    assert "Conversion reverted" in info.get("reason", "") or "Not classified" in info.get("reason", "")
    # Original values should remain text
    assert list(converted.astype(str)) == list(series.astype(str))
