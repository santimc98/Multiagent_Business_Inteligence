import json
import math
from datetime import date, datetime
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in runtime
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency in runtime
    pd = None


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return [to_jsonable(item) for item in value.tolist()]
    if pd is not None:
        if value is pd.NA:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Series):
            return [to_jsonable(item) for item in value.tolist()]
        if isinstance(value, pd.Index):
            return [to_jsonable(item) for item in value.tolist()]
        if isinstance(value, pd.DataFrame):
            return {str(key): to_jsonable(val) for key, val in value.to_dict(orient="list").items()}
        try:
            if pd.isna(value) is True:
                return None
        except Exception:
            pass
    return str(value)


def dump_json(path: str, obj: Any) -> None:
    payload = to_jsonable(obj)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
