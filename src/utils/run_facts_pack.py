import json
from typing import Any, Dict, List, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _unique_strings(items: List[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_contract(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = (
        state.get("execution_contract")
        or state.get("execution_contract_min")
        or state.get("contract_min")
        or {}
    )
    return contract if isinstance(contract, dict) else {}


def _extract_objective_type(contract: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    eval_spec = _as_dict(contract.get("evaluation_spec"))
    obj_analysis = _as_dict(contract.get("objective_analysis"))
    strategy_spec = _as_dict(state.get("strategy_spec"))
    return (
        eval_spec.get("objective_type")
        or obj_analysis.get("problem_type")
        or strategy_spec.get("objective_type")
        or contract.get("objective_type")
    )


def _extract_target_columns(contract: Dict[str, Any]) -> List[str]:
    targets: List[Any] = []
    column_roles = _as_dict(contract.get("column_roles"))
    targets.extend(_as_list(column_roles.get("outcome")))
    if contract.get("target_column"):
        targets.append(contract.get("target_column"))
    targets.extend(_as_list(contract.get("target_columns")))
    eval_spec = _as_dict(contract.get("evaluation_spec"))
    if eval_spec.get("target_column"):
        targets.append(eval_spec.get("target_column"))
    return _unique_strings(targets)


def _extract_required_outputs(contract: Dict[str, Any]) -> List[str]:
    required = _as_list(contract.get("required_outputs"))
    if required:
        return _unique_strings(required)
    artifact_reqs = _as_dict(contract.get("artifact_requirements"))
    files = _as_list(artifact_reqs.get("required_files"))
    return _unique_strings(files)


def _extract_columns_sample(state: Dict[str, Any], max_cols: int = 40) -> Dict[str, Any]:
    columns = (
        state.get("column_inventory")
        or state.get("cleaned_column_inventory")
        or _as_dict(state.get("ml_context_snapshot")).get("cleaned_column_inventory")
        or []
    )
    if isinstance(columns, dict):
        columns_list = [str(c) for c in _as_list(columns.get("columns")) if c]
    else:
        columns_list = [str(c) for c in _as_list(columns) if c]
    return {
        "n_cols": len(columns_list) if columns_list else None,
        "sample": columns_list[:max_cols],
    }


def build_run_facts_pack(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, deterministic snapshot of run facts for agent context.
    """
    state = state or {}
    contract = _extract_contract(state)
    dialect = {
        "sep": state.get("csv_sep"),
        "decimal": state.get("csv_decimal"),
        "encoding": state.get("csv_encoding"),
    }
    column_info = _extract_columns_sample(state)
    dataset_scale_hints = state.get("dataset_scale_hints")
    if not isinstance(dataset_scale_hints, dict):
        dataset_scale_hints = {}

    run_facts = {
        "run_id": state.get("run_id"),
        "input_csv_path": state.get("csv_path"),
        "dialect": dialect,
        "dataset_scale_hints": dataset_scale_hints or None,
        "objective_type": _extract_objective_type(contract, state),
        "target_columns": _extract_target_columns(contract),
        "required_outputs": _extract_required_outputs(contract),
        "visual_requirements": _as_dict(_as_dict(contract.get("artifact_requirements")).get("visual_requirements"))
        or None,
        "decisioning_requirements": _as_dict(contract.get("decisioning_requirements")) or None,
        "column_inventory": column_info,
        "iteration": {
            "iteration_count": state.get("iteration_count"),
            "data_engineer_attempt": state.get("data_engineer_attempt"),
            "ml_engineer_attempt": state.get("ml_engineer_attempt"),
            "reviewer_iteration": state.get("reviewer_iteration"),
        },
    }
    return run_facts


def format_run_facts_block(run_facts: Dict[str, Any], max_chars: int = 4000) -> str:
    payload = json.dumps(run_facts or {}, indent=2, sort_keys=True, ensure_ascii=True)
    block = (
        "=== RUN_FACTS_PACK_JSON (read-only) ===\n"
        + payload
        + "\n=== END RUN_FACTS_PACK_JSON ==="
    )
    if len(block) <= max_chars:
        return block

    head_len = max(0, int(max_chars * 0.6) - 20)
    tail_len = max(0, max_chars - head_len - 30)
    head = block[:head_len]
    tail = block[-tail_len:] if tail_len else ""
    return f"{head}\n...(truncated)...\n{tail}"
