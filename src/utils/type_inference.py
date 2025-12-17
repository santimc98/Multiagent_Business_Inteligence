import re
from typing import Dict, Tuple, Optional

import pandas as pd
from src.utils.number_parsing import parse_localized_number


def profile_numeric_currency_candidate(
    series: pd.Series,
    sample_size: int = 200,
    digit_frac_threshold: float = 0.6,
    parse_success_threshold: float = 0.9,
    decimal_hint: Optional[str] = None,
    thousands_hint: Optional[str] = None,
) -> Dict[str, float]:
    non_null = series.dropna()
    total_non_null = len(non_null)
    if total_non_null == 0:
        return {
            "digits_ratio": 0.0,
            "parse_success_rate": 0.0,
            "total_non_null": 0,
            "sample_size": 0,
            "is_numeric_currency": False,
        }

    str_vals = non_null.astype(str)
    digits_hits = sum(bool(re.search(r"\d", v)) for v in str_vals)
    digits_ratio = digits_hits / total_non_null

    sample = str_vals.sample(min(sample_size, total_non_null), random_state=0)
    parsed_sample = sample.apply(
        lambda x: parse_localized_number(x, decimal_hint=decimal_hint, thousands_hint=thousands_hint)
    )
    sample_len = len(parsed_sample)
    parse_success_rate = parsed_sample.notna().sum() / sample_len if sample_len else 0.0

    is_mostly_alpha = digits_ratio < digit_frac_threshold
    is_numeric_currency = (
        digits_ratio >= digit_frac_threshold and parse_success_rate >= parse_success_threshold and not is_mostly_alpha
    )

    return {
        "digits_ratio": digits_ratio,
        "parse_success_rate": parse_success_rate,
        "total_non_null": total_non_null,
        "sample_size": sample_len,
        "is_numeric_currency": is_numeric_currency,
    }


def safe_convert_numeric_currency(
    series: pd.Series,
    sample_size: int = 200,
    digit_frac_threshold: float = 0.6,
    parse_success_threshold: float = 0.9,
    post_drop_threshold: float = 0.7,
    decimal_hint: Optional[str] = None,
    thousands_hint: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Converts a series to numeric currency with guardrails.
    Reverts to text if parse success is low or conversion wipes data.
    """
    profile = profile_numeric_currency_candidate(
        series,
        sample_size=sample_size,
        digit_frac_threshold=digit_frac_threshold,
        parse_success_threshold=parse_success_threshold,
        decimal_hint=decimal_hint,
        thousands_hint=thousands_hint,
    )

    info: Dict[str, object] = {
        "parse_success_rate": profile["parse_success_rate"],
        "digits_ratio": profile["digits_ratio"],
        "reverted": False,
        "reason": "",
        "decimal_hint": decimal_hint,
        "thousands_hint": thousands_hint,
        "sample_size": profile.get("sample_size", 0),
        "total_non_null": profile.get("total_non_null", 0),
    }

    if not profile["is_numeric_currency"]:
        info["reverted"] = True
        info["reason"] = "Not classified as numeric_currency by profile thresholds."
        return series.astype("string"), info

    non_null_before = series.dropna().shape[0]
    def _parse_full(x):
        return parse_localized_number(x, decimal_hint=decimal_hint, thousands_hint=thousands_hint)

    converted = series.astype(str).apply(_parse_full)
    converted_numeric = pd.to_numeric(converted, errors="coerce")
    non_null_after = converted_numeric.dropna().shape[0]
    after_rate = non_null_after / non_null_before if non_null_before else 0.0

    if non_null_before > 0 and (non_null_after == 0 or after_rate < post_drop_threshold):
        info["reverted"] = True
        info["reason"] = f"Conversion reverted due to low parse success (after_rate={after_rate:.3f})."
        return series.astype("string"), info

    info["after_rate"] = after_rate
    return converted_numeric.astype(float), info
