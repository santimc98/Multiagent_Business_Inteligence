from src.utils.number_parsing import parse_localized_number
from src.utils.type_inference import safe_convert_numeric_currency
import pandas as pd


def test_parse_eu_number():
    assert parse_localized_number("€ 11.883.078", decimal_hint=",") == 11883078.0
    assert parse_localized_number("€ 8.004", decimal_hint=",") == 8004.0
    assert parse_localized_number("...€ 8.004", decimal_hint=",") == 8004.0


def test_parse_dot_decimal_respected():
    assert parse_localized_number("11.5", decimal_hint=".") == 11.5


def test_guardrail_reverts_on_all_nan():
    series = pd.Series(["abc", "def", "ghi"])
    converted, info = safe_convert_numeric_currency(series, parse_success_threshold=0.9, post_drop_threshold=0.7, decimal_hint=",")
    assert info["reverted"] is True
    assert list(converted.astype(str)) == list(series.astype(str))
