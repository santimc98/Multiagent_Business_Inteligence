import os
from typing import Any, Dict, Optional


def _resolve_csv_path_with_base(base_cwd: str, csv_path: str) -> str:
    if not csv_path:
        return csv_path
    if os.path.isabs(csv_path):
        return os.path.normpath(csv_path)
    if base_cwd:
        return os.path.normpath(os.path.abspath(os.path.join(base_cwd, csv_path)))
    return os.path.normpath(os.path.abspath(csv_path))


def _add_workspace_metadata(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    orig_cwd_pre: Optional[str],
    run_dir: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "csv_path": state.get("csv_path"),
            "_orig_cwd": state.get("_orig_cwd") or orig_cwd_pre,
            "work_dir": state.get("work_dir"),
            "workspace_active": state.get("workspace_active", True),
        }
    )
    if run_dir:
        payload["run_dir"] = run_dir
    return payload
