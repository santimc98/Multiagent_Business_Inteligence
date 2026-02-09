"""
Run workspace isolation utilities.

Ensures each run operates in an isolated workspace directory to prevent
cross-run contamination from leftover artifacts.
"""
import os
import re
from typing import Dict, Any, Optional


_RUN_WORKSPACE_RE = re.compile(
    r"^(?P<root>.+?)[\\/]+runs[\\/]+[^\\/]+[\\/]+work(?:[\\/].*)?$",
    re.IGNORECASE,
)


def _infer_project_root_from_workspace_path(path: Optional[str]) -> Optional[str]:
    """
    Infer project root when current cwd is inside runs/<run_id>/work.
    """
    if not path:
        return None
    normalized = os.path.normpath(os.path.abspath(path))
    match = _RUN_WORKSPACE_RE.match(normalized)
    if not match:
        return None
    root = match.group("root")
    if root and os.path.isdir(root):
        return os.path.normpath(root)
    return None


def recover_orphaned_workspace_cwd(project_root: Optional[str] = None) -> Optional[str]:
    """
    Recover cwd if process got stranded inside runs/<run_id>/work.

    Returns restored path when recovery happens, else None.
    """
    cwd = os.getcwd()
    inferred_root = _infer_project_root_from_workspace_path(cwd)
    if not inferred_root:
        return None

    candidate = None
    if project_root:
        normalized_root = os.path.normpath(os.path.abspath(project_root))
        if os.path.isdir(normalized_root):
            candidate = normalized_root
    if not candidate:
        candidate = inferred_root

    if os.path.normcase(os.path.normpath(cwd)) != os.path.normcase(candidate):
        os.chdir(candidate)
        print(f"WORKSPACE_RECOVER: Restored cwd to {candidate}")
    return candidate


def init_run_workspace(run_dir: str) -> str:
    """
    Initialize a workspace directory for a run.

    Creates work_dir with required subdirectories for artifacts isolation.

    Args:
        run_dir: The run bundle directory (e.g., runs/run_xxx)

    Returns:
        work_dir: Path to the workspace directory (run_dir/work)
    """
    work_dir = os.path.join(run_dir, "work")

    # Create workspace and all required subdirectories
    subdirs = [
        "data",
        "reports",
        "static/plots",
        "analysis",
        "models",
        "plots",
        "artifacts",
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(work_dir, subdir), exist_ok=True)

    return work_dir


def enter_run_workspace(state: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    """
    Enter the run workspace - changes cwd to isolated workspace.

    This ensures all relative paths (data/, reports/, etc.) resolve
    to the run-specific workspace, not the global root.

    Args:
        state: Agent state dict
        run_dir: The run bundle directory

    Returns:
        Updated state with workspace info
    """
    # Save original cwd for restoration
    orig_cwd = os.getcwd()
    state["orig_cwd"] = orig_cwd

    for key in ("csv_path", "raw_csv_path", "input_csv_path"):
        value = state.get(key)
        if not value or not isinstance(value, str):
            continue
        if os.path.isabs(value):
            continue
        patched = os.path.normpath(os.path.abspath(os.path.join(orig_cwd, value)))
        if patched != value:
            print(f"WORKSPACE_PATH_PATCH: {key} '{value}' -> '{patched}'")
        state[key] = patched
        if not os.path.exists(patched):
            print(f"WORKSPACE_PATH_MISSING: {key} '{patched}'")

    # Initialize and enter workspace
    work_dir = init_run_workspace(run_dir)
    state["work_dir"] = work_dir
    state["workspace_active"] = True

    # Change to workspace directory
    os.chdir(work_dir)

    # Double-check required dirs exist (defense in depth)
    for subdir in ["data", "reports", "static/plots"]:
        os.makedirs(subdir, exist_ok=True)

    print(f"WORKSPACE_ENTER: Entered run workspace at {work_dir}")

    return state


def exit_run_workspace(state: Dict[str, Any]) -> None:
    """
    Exit the run workspace - restores original cwd.

    Args:
        state: Agent state dict with orig_cwd
    """
    orig_cwd = state.get("orig_cwd") if isinstance(state, dict) else None
    restored = False
    if orig_cwd and os.path.isdir(orig_cwd):
        os.chdir(orig_cwd)
        print(f"WORKSPACE_EXIT: Restored cwd to {orig_cwd}")
        restored = True

    if not restored:
        recovered = recover_orphaned_workspace_cwd()
        if recovered:
            print(f"WORKSPACE_EXIT_FALLBACK: Restored cwd to {recovered}")

    if isinstance(state, dict):
        state["workspace_active"] = False

    # Note: We don't delete work_dir by default (useful for debug).
    # Set env CLEANUP_RUN_WORKSPACE=1 to enable cleanup in future.


def get_work_dir(state: Dict[str, Any]) -> Optional[str]:
    """
    Get the current work directory from state.

    Returns:
        work_dir if workspace is active, None otherwise
    """
    if state.get("workspace_active"):
        return state.get("work_dir")
    return None


def is_workspace_active(state: Dict[str, Any]) -> bool:
    """Check if we're currently in a run workspace."""
    return bool(state.get("workspace_active"))
