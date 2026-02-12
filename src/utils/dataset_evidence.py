import csv
import random
from collections import deque
from typing import Any, Dict, List

import pandas as pd

_NULL_STRINGS = {"", "na", "n/a", "nan", "null", "none", "nat"}


def read_header(csv_path: str, dialect: Dict[str, Any]) -> List[str]:
    if not csv_path:
        return []
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    try:
        header_df = pd.read_csv(csv_path, nrows=0, sep=sep, decimal=decimal, encoding=encoding)
        return [str(col) for col in header_df.columns]
    except Exception:
        pass
    try:
        with open(csv_path, "r", encoding=encoding, errors="replace") as handle:
            reader = csv.reader(handle, delimiter=sep)
            header = next(reader, [])
        return [str(col) for col in header]
    except Exception:
        return []


def scan_missingness(
    csv_path: str,
    dialect: Dict[str, Any],
    col: str,
    chunksize: int = 200000,
) -> Dict[str, Any]:
    if not csv_path or not col:
        return {"column": col, "total": 0, "missing": 0, "null_frac_exact": None}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    total = 0
    missing = 0
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=[col],
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        )
    except Exception:
        return {"column": col, "total": 0, "missing": 0, "null_frac_exact": None}

    for chunk in reader:
        if col not in chunk.columns:
            continue
        series = chunk[col]
        total += int(series.shape[0])
        cleaned = series.astype("string").str.strip()
        lowered = cleaned.str.lower()
        missing_mask = series.isna() | (lowered == "") | lowered.isin(_NULL_STRINGS)
        missing += int(missing_mask.sum())
    null_frac_exact = float(missing / total) if total else None
    return {"column": col, "total": int(total), "missing": int(missing), "null_frac_exact": null_frac_exact}


def scan_uniques(
    csv_path: str,
    dialect: Dict[str, Any],
    col: str,
    chunksize: int = 200000,
    max_unique: int = 20,
) -> Dict[str, Any]:
    if not csv_path or not col:
        return {"column": col, "unique_values": [], "counts_hint": [], "total": 0}
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    total = 0
    counts: Dict[str, int] = {}
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=[col],
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        )
    except Exception:
        return {"column": col, "unique_values": [], "counts_hint": [], "total": 0}

    for chunk in reader:
        if col not in chunk.columns:
            continue
        series = chunk[col]
        cleaned = series.astype("string").str.strip()
        lowered = cleaned.str.lower()
        missing_mask = series.isna() | (lowered == "") | lowered.isin(_NULL_STRINGS)
        values = cleaned[~missing_mask]
        total += int(values.shape[0])
        for raw_val in values.tolist():
            val = str(raw_val)
            if val in counts:
                counts[val] += 1
            elif len(counts) < max_unique:
                counts[val] = 1

    counts_hint = [
        {"value": key, "count": int(count)} for key, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
    unique_values = [entry["value"] for entry in counts_hint]
    return {"column": col, "unique_values": unique_values, "counts_hint": counts_hint, "total": int(total)}


def _truncate_value(value: Any, max_chars: int) -> Any:
    if value is None:
        return None
    text = str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _select_sample_columns(columns: List[str], max_columns: int) -> List[str]:
    if not isinstance(columns, list):
        return []
    normalized = [str(col) for col in columns if col]
    if max_columns <= 0 or len(normalized) <= max_columns:
        return normalized

    head_n = min(8, max_columns)
    tail_n = min(4, max(0, max_columns - head_n))
    selected_idx = set(range(head_n))
    if tail_n > 0:
        selected_idx.update(range(max(0, len(normalized) - tail_n), len(normalized)))

    remaining = max_columns - len(selected_idx)
    if remaining > 0:
        middle_start = head_n
        middle_end = max(0, len(normalized) - tail_n)
        middle_indices = list(range(middle_start, middle_end))
        if middle_indices:
            step = max(1, len(middle_indices) // remaining)
            for idx in middle_indices[::step]:
                selected_idx.add(idx)
                if len(selected_idx) >= max_columns:
                    break

    ordered = [normalized[idx] for idx in range(len(normalized)) if idx in selected_idx]
    return ordered[:max_columns]


def _compact_record(row: Dict[str, Any], keep_columns: List[str], max_value_chars: int) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in keep_columns:
        if key not in row:
            continue
        compact[str(key)] = _truncate_value(row.get(key), max_value_chars)
    return compact


def sample_rows(
    csv_path: str,
    dialect: Dict[str, Any],
    head_n: int = 50,
    tail_n: int = 50,
    random_n: int = 50,
    seed: int = 42,
    max_columns: int = 40,
    max_value_chars: int = 120,
) -> Dict[str, Any]:
    if not csv_path:
        return {
            "head": [],
            "tail": [],
            "random": [],
            "total_rows": 0,
            "sampled_columns": [],
            "sampled_column_count": 0,
            "column_sampling_applied": False,
            "value_char_limit": max_value_chars,
        }
    sep = dialect.get("sep") or ","
    decimal = dialect.get("decimal") or "."
    encoding = dialect.get("encoding") or "utf-8"
    all_columns = read_header(csv_path, dialect)
    sampled_columns = _select_sample_columns(all_columns, max_columns=max_columns)
    usecols = sampled_columns if sampled_columns else None
    head: List[Dict[str, Any]] = []
    tail: deque[Dict[str, Any]] = deque(maxlen=max(0, int(tail_n)))
    reservoir: List[Dict[str, Any]] = []
    total_rows = 0
    rng = random.Random(seed)
    try:
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            encoding=encoding,
            dtype="string",
            keep_default_na=False,
            usecols=usecols,
            chunksize=5000,
            low_memory=False,
        )
    except Exception:
        return {
            "head": [],
            "tail": [],
            "random": [],
            "total_rows": 0,
            "sampled_columns": sampled_columns,
            "sampled_column_count": len(sampled_columns),
            "column_sampling_applied": len(sampled_columns) < len(all_columns),
            "value_char_limit": max_value_chars,
        }

    for chunk in reader:
        records = chunk.to_dict(orient="records")
        for row in records:
            compact_row = _compact_record(
                row,
                sampled_columns if sampled_columns else [str(col) for col in row.keys()],
                max_value_chars=max_value_chars,
            )
            total_rows += 1
            if len(head) < head_n:
                head.append(compact_row)
            if tail_n > 0:
                tail.append(compact_row)
            if random_n > 0:
                if len(reservoir) < random_n:
                    reservoir.append(compact_row)
                else:
                    idx = rng.randrange(total_rows)
                    if idx < random_n:
                        reservoir[idx] = compact_row

    return {
        "head": head,
        "tail": list(tail),
        "random": reservoir,
        "total_rows": total_rows,
        "sampled_columns": sampled_columns,
        "sampled_column_count": len(sampled_columns),
        "total_column_count": len(all_columns),
        "column_sampling_applied": len(sampled_columns) < len(all_columns),
        "value_char_limit": max_value_chars,
    }
