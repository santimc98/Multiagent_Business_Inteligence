"""
Data Profile Preflight Module.

Ensures data_profile.json artifact is created early in the pipeline,
BEFORE the Data Engineer runs, so it's always available for QA/audit
even if the pipeline aborts.

This is a "belt & suspenders" approach:
  1) After execution_contract.json is persisted (run_execution_planner)
  2) At start of run_data_engineer (backup)
  3) At start of run_ml_engineer (final fallback)
"""
import json
import os
from typing import Dict, Any, Tuple, List, Optional

from src.utils.json_sanitize import dump_json


def _abs_in_work(work_dir_abs: str, rel_path: str) -> str:
    """Return absolute path inside work directory."""
    if not work_dir_abs:
        return rel_path
    return os.path.abspath(os.path.join(work_dir_abs, rel_path))


def _load_json_safe(path: str) -> Dict[str, Any]:
    """Load JSON file safely, return {} on error."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def ensure_data_profile_artifact(
    state: Dict[str, Any],
    contract: Dict[str, Any],
    analysis_type: Optional[str],
    work_dir_abs: str,
    run_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Ensure data_profile.json artifact exists early in the pipeline.

    This function is idempotent: if data_profile already exists in state
    and on disk, it returns early without rewriting.

    Priority:
      1) If state["data_profile"] exists and file exists -> return early
      2) Try to convert from dataset_profile.json (best quality)
      3) Fall back to minimal profile (so QA doesn't crash)

    Args:
        state: Pipeline state dict (will be mutated with data_profile keys)
        contract: Execution contract (for outcome columns, etc.)
        analysis_type: Optional analysis type (classification/regression)
        work_dir_abs: Absolute path to workspace root
        run_id: Optional run ID for logging

    Returns:
        Tuple of (data_profile or None, source string)
        source is one of: "already_present", "converted_from_dataset_profile",
                          "minimal_fallback", "failed"
    """
    from src.utils.run_logger import log_run_event

    # Check if already present
    existing_profile = state.get("data_profile")
    existing_path = state.get("data_profile_path")
    if existing_profile and isinstance(existing_profile, dict):
        # Verify file exists
        if existing_path:
            full_path = _abs_in_work(work_dir_abs, existing_path)
            if os.path.exists(full_path):
                return existing_profile, "already_present"
        # Profile in state but file missing - re-write it
        if existing_profile.get("basic_stats"):
            try:
                from src.agents.steward import write_data_profile
                output_path = _abs_in_work(work_dir_abs, "data/data_profile.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_data_profile(existing_profile, output_path)
                state["data_profile_path"] = "data/data_profile.json"
                return existing_profile, "already_present"
            except Exception:
                pass

    try:
        from src.utils.data_profile_compact import convert_dataset_profile_to_data_profile
        from src.agents.steward import write_data_profile
        from src.utils.contract_v41 import get_outcome_columns

        data_profile = None
        profile_source = "failed"

        # Try #1: Load and convert dataset_profile.json
        dataset_profile_path = _abs_in_work(work_dir_abs, "data/dataset_profile.json")
        dataset_profile = _load_json_safe(dataset_profile_path)

        if (
            dataset_profile
            and isinstance(dataset_profile, dict)
            and dataset_profile.get("rows")
            and dataset_profile.get("cols")
            and isinstance(dataset_profile.get("missing_frac"), dict)
        ):
            # Valid dataset_profile found - convert it
            data_profile = convert_dataset_profile_to_data_profile(
                dataset_profile, contract or {}, analysis_type
            )
            profile_source = "converted_from_dataset_profile"
            print(f"DATA_PROFILE_PREFLIGHT: Converted from {dataset_profile_path}")

        # Try #2: Minimal fallback if no dataset_profile
        if not data_profile:
            # Build minimal profile from available information
            columns: List[str] = []

            # Prefer column_inventory from state
            col_inv = state.get("column_inventory")
            if isinstance(col_inv, dict):
                columns = col_inv.get("columns", [])
            elif isinstance(col_inv, list):
                columns = col_inv

            # Fallback to contract canonical_columns
            if not columns and isinstance(contract, dict):
                canonical = contract.get("canonical_columns")
                if isinstance(canonical, list):
                    columns = [str(c) for c in canonical if c]

            n_cols = len(columns)
            n_rows = 0  # Unknown

            # Try to get outcome columns
            outcome_analysis = {}
            try:
                outcome_cols = get_outcome_columns(contract or {})
                for oc in outcome_cols:
                    outcome_analysis[oc] = {"present": "unknown", "error": "no_dataset_profile"}
            except Exception:
                pass

            data_profile = {
                "basic_stats": {
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "columns": columns,
                },
                "dtypes": {},
                "missingness_top30": {},
                "outcome_analysis": outcome_analysis,
                "split_candidates": [],
                "constant_columns": [],
                "high_cardinality_columns": [],
                "leakage_flags": [],
                "schema_version": "1.0",
                "source": "minimal_fallback_no_dataset_profile",
            }
            profile_source = "minimal_fallback"
            print(f"DATA_PROFILE_PREFLIGHT: Created minimal fallback (no dataset_profile.json)")

        # Write the profile
        if data_profile:
            output_path = _abs_in_work(work_dir_abs, "data/data_profile.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_data_profile(data_profile, output_path)

            # Update state
            state["data_profile"] = data_profile
            state["data_profile_path"] = "data/data_profile.json"
            state["data_profile_source"] = profile_source

            # Log event
            if run_id:
                try:
                    log_run_event(run_id, "data_profile_built", {
                        "source": profile_source,
                        "n_rows": data_profile.get("basic_stats", {}).get("n_rows"),
                        "n_cols": data_profile.get("basic_stats", {}).get("n_cols"),
                        "outcome_cols": list(data_profile.get("outcome_analysis", {}).keys()),
                    })
                except Exception:
                    pass

            return data_profile, profile_source

    except Exception as err:
        print(f"DATA_PROFILE_PREFLIGHT: Failed - {err}")
        if run_id:
            try:
                from src.utils.run_logger import log_run_event
                log_run_event(run_id, "data_profile_failed", {"error": str(err)})
            except Exception:
                pass
        return None, "failed"

    return None, "failed"
