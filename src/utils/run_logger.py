import json
import os
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from src.utils.context_pack import compress_long_lists
from src.utils.text_encoding import sanitize_text_payload_with_stats

LOG_PATHS: Dict[str, str] = {}


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def register_run_log(run_id: str, path: str) -> None:
    # Store absolute path so logging does not depend on cwd.
    LOG_PATHS[run_id] = os.path.abspath(path)


def get_log_path(run_id: str, log_dir: str = "logs") -> str:
    if run_id in LOG_PATHS:
        return LOG_PATHS[run_id]
    # Default path is absolute as well.
    return os.path.abspath(os.path.join(log_dir, f"run_{run_id}.jsonl"))


def init_run_log(run_id: str, metadata: Optional[Dict[str, Any]] = None, log_dir: str = "logs") -> str:
    path = get_log_path(run_id, log_dir=log_dir)
    _ensure_parent_dir(path)

    # Create file if missing.
    with open(path, "a", encoding="utf-8") as _:
        pass

    if metadata is not None:
        log_run_event(run_id, "run_init", metadata, log_dir=log_dir)

    return path


def log_run_event(run_id: str, event_type: str, payload: Dict[str, Any], log_dir: str = "logs") -> None:
    path = get_log_path(run_id, log_dir=log_dir)
    _ensure_parent_dir(path)
    safe_payload = payload
    try:
        safe_payload, _ = compress_long_lists(payload)
    except Exception:
        safe_payload = payload

    payload_sanitized, encoding_stats = sanitize_text_payload_with_stats(safe_payload)
    event: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event_type,
        "payload": payload_sanitized,
    }
    if encoding_stats.get("strings_changed", 0) > 0:
        event["encoding_guard"] = {
            "strings_changed": int(encoding_stats.get("strings_changed", 0)),
            "mojibake_hits": int(encoding_stats.get("mojibake_hits", 0)),
        }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def finalize_run_log(
    run_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    log_dir: str = "logs",
) -> str:
    """
    Backward-compatible entrypoint used by other modules.
    Ensures the log exists and records a final event.
    """
    path = init_run_log(run_id, metadata=None, log_dir=log_dir)

    if metadata is None:
        metadata = {}

    # Final marker event.
    log_run_event(run_id, "run_finalize", metadata, log_dir=log_dir)
    return path
