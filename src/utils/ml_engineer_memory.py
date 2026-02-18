import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _memory_path(run_id: str, base_dir: str = "runs") -> Path:
    run_token = str(run_id or "").strip()
    if not run_token:
        raise ValueError("run_id is required")
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = _project_root() / base_path
    return base_path / run_token / "work" / "memory" / "ml_engineer_memory.jsonl"


def append_memory(run_id: str, entry_dict: Dict[str, Any], base_dir: str = "runs") -> str | None:
    if not run_id or not isinstance(entry_dict, dict):
        return None
    try:
        path = _memory_path(run_id, base_dir=base_dir)
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


def load_recent_memory(run_id: str, k: int = 5, base_dir: str = "runs") -> List[Dict[str, Any]]:
    if not run_id:
        return []
    try:
        path = _memory_path(run_id, base_dir=base_dir)
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
    count = int(k or 0)
    if count <= 0:
        return []
    return entries[-count:]
