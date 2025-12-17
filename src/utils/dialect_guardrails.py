import json
from typing import Any, Dict, Tuple

import pandas as pd


def get_output_dialect_from_manifest(
    manifest_path: str,
    default_sep: str,
    default_decimal: str,
    default_encoding: str,
) -> Tuple[str, str, str, bool]:
    """
    Loads output_dialect from a cleaning manifest if available.
    Returns (sep, decimal, encoding, updated_flag).
    Falls back to provided defaults when manifest or keys are missing.
    """
    sep = default_sep
    decimal = default_decimal
    encoding = default_encoding
    updated = False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        output = manifest.get("output_dialect") if isinstance(manifest, dict) else None
        if isinstance(output, dict):
            sep = output.get("sep", sep)
            decimal = output.get("decimal", decimal)
            encoding = output.get("encoding", encoding)
            updated = True
    except Exception:
        # Swallow errors to keep behavior permissive; callers decide next steps.
        updated = False

    return sep, decimal, encoding, updated


def assert_not_single_column_delimiter_mismatch(
    df: pd.DataFrame, used_sep: str, used_decimal: str, used_encoding: str
) -> None:
    """
    Raises ValueError if the dataframe looks like a mis-parsed CSV
    (single column whose header still contains a delimiter).
    Guardrails avoid false positives with a length threshold.
    """
    if df.shape[1] != 1:
        return

    colname = str(df.columns[0])
    if len(colname) <= 20:
        return

    if any(delim in colname for delim in [",", ";", "\t"]):
        raise ValueError(
            "Delimiter/Dialect mismatch: loaded a single-column frame with delimiter-like header; "
            f"used sep='{used_sep}', decimal='{used_decimal}', encoding='{used_encoding}'. "
            "Likely delimiter mismatch between output_dialect and actual cleaned CSV serialization."
        )
