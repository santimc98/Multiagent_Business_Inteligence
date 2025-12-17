import pandas as pd


def is_effectively_missing(value) -> bool:
    """
    Returns True for values that should be treated as missing:
    - None / NaN
    - Empty strings or whitespace-only
    - DOES NOT treat 0 or "0" as missing
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        if value.strip() == "":
            return True
        if value.strip() == "0":
            return False
    if value == 0:
        return False
    return False


def is_effectively_missing_series(series: pd.Series) -> pd.Series:
    return series.apply(is_effectively_missing)
