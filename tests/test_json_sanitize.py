import json

import numpy as np
import pandas as pd

from src.utils.json_sanitize import dump_json, to_jsonable


def test_to_jsonable_handles_numpy_and_pandas():
    payload = {
        "flag": np.bool_(True),
        "count": np.int64(7),
        "ts": pd.Timestamp("2025-01-01T00:00:00Z"),
        "arr": np.array([1, 2, 3]),
    }
    result = to_jsonable(payload)
    assert result["flag"] is True
    assert result["count"] == 7
    assert result["ts"].startswith("2025-01-01")
    assert result["arr"] == [1, 2, 3]


def test_dump_json_writes_safely(tmp_path):
    path = tmp_path / "payload.json"
    payload = {"ok": np.bool_(False), "when": pd.Timestamp("2025-01-02")}
    dump_json(str(path), payload)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["ok"] is False
    assert "2025-01-02" in parsed["when"]
