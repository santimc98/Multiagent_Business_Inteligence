from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tracker_path(run_id: str, base_dir: str = "runs") -> Path:
    token = str(run_id or "").strip()
    if not token:
        raise ValueError("run_id is required")
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = _project_root() / base_path
    return base_path / token / "work" / "memory" / "experiment_tracker.jsonl"


def build_hypothesis_signature(
    *,
    technique: Any,
    target_columns: Any,
    feature_scope: Any,
    params: Any,
) -> str:
    tech = str(technique or "").strip().lower() or "unknown_technique"
    scope = str(feature_scope or "").strip().lower() or "model_features"
    if isinstance(target_columns, list):
        cols = sorted(
            [
                str(item).strip()
                for item in target_columns
                if str(item or "").strip()
            ]
        )
    else:
        col = str(target_columns or "").strip()
        cols = [col] if col else []
    params_payload = params if isinstance(params, dict) else {}
    params_json = json.dumps(params_payload, ensure_ascii=True, sort_keys=True)
    signature_base = "technique={};scope={};cols={};params={}".format(
        tech,
        scope,
        ",".join(cols),
        params_json,
    )
    digest = hashlib.sha1(signature_base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return "hyp_" + digest


def append_experiment_entry(run_id: str, entry_dict: Dict[str, Any], base_dir: str = "runs") -> str | None:
    if not run_id or not isinstance(entry_dict, dict):
        return None
    try:
        path = _tracker_path(run_id, base_dir=base_dir)
    except Exception:
        return None
    payload = dict(entry_dict)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return str(path)
    except Exception:
        return None


def load_recent_experiment_entries(run_id: str, k: int = 20, base_dir: str = "runs") -> List[Dict[str, Any]]:
    if not run_id:
        return []
    try:
        path = _tracker_path(run_id, base_dir=base_dir)
    except Exception:
        return []
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except Exception:
        return []
    limit = int(k or 0)
    if limit <= 0:
        return []
    return entries[-limit:]

