import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import Iterable, Optional, Set


def _looks_like_id(series: pd.Series) -> bool:
    if series.empty:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    nunique = non_null.nunique()
    ratio = nunique / len(non_null)
    if ratio < 0.9:
        return False
    # Prefer string-like or integer-ish identifiers
    if series.dtype == object:
        return True
    if is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Avoid continuous floats; require integer-like values
        if pd.api.types.is_integer_dtype(series):
            return True
        # For float, check if values are near integers
        rounded = non_null.round().astype("Int64")
        if (non_null - rounded).abs().max() < 1e-6:
            return True
    return False


def infer_group_key(df: pd.DataFrame, exclude_cols: Iterable[str] = ()) -> Optional[pd.Series]:
    """
    Infers a grouping key to avoid leakage in validation splits.
    - Prefer an explicit ID-like column (high cardinality, mostly unique).
    - Fallback: hash rows of X (excluding target/exclude_cols) to create stable groups.
    Returns a Series of group labels or None if not enough info.
    """
    exclude: Set[str] = set(exclude_cols or [])
    candidate_cols = [c for c in df.columns if c not in exclude]

    for col in candidate_cols:
        series = df[col]
        if _looks_like_id(series):
            safe = series.fillna("__missing_id__").astype(str)
            return pd.util.hash_pandas_object(safe, index=False)

    # Fallback to hashed groups using feature rows
    feature_df = df.drop(columns=list(exclude), errors="ignore")
    if feature_df.empty:
        return None
    hashes = pd.util.hash_pandas_object(feature_df, index=False)
    if hashes.nunique() <= 1:
        return None
    return hashes
