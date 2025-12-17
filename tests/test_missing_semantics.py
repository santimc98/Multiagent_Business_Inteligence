import pandas as pd

from src.utils.missing import is_effectively_missing, is_effectively_missing_series


def test_missing_semantics_zero_not_missing():
    assert is_effectively_missing(0) is False
    assert is_effectively_missing("0") is False


def test_missing_semantics_empty_missing():
    assert is_effectively_missing("") is True
    assert is_effectively_missing("   ") is True
    s = pd.Series(["", "0", None, "  "])
    res = is_effectively_missing_series(s)
    assert res.tolist() == [True, False, True, True]


def test_missing_semantics_pd_na_missing():
    assert is_effectively_missing(pd.NA) is True
