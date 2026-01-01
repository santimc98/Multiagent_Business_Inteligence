import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.run_logger import register_run_log

RUNS_DIR = "runs"

_RUN_DIRS: Dict[str, str] = {}
_RUN_ATTEMPTS: Dict[str, List[Dict[str, Any]]] = {}
_TEE_STREAM = None


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _hash_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _normalize_required_outputs(contract: Dict[str, Any]) -> List[str]:
    if not isinstance(contract, dict):
        return []
    spec = contract.get("spec_extraction") or {}
    deliverables = spec.get("deliverables") if isinstance(spec, dict) else None
    if isinstance(deliverables, list) and deliverables:
        required = []
        for item in deliverables:
            if isinstance(item, dict):
                if item.get("required") and item.get("path"):
                    required.append(str(item.get("path")))
            elif isinstance(item, str):
                required.append(item)
        return required
    return contract.get("required_outputs", []) or []


def _normalize_produced_outputs(artifact_index: Any) -> List[str]:
    produced: List[str] = []
    if isinstance(artifact_index, list):
        for item in artifact_index:
            if isinstance(item, dict):
                path = item.get("path")
                if path:
                    produced.append(str(path))
            elif isinstance(item, str):
                produced.append(item)
    return produced


def init_run_bundle(
    run_id: str,
    state: Optional[Dict[str, Any]] = None,
    base_dir: str = RUNS_DIR,
    enable_tee: bool = True,
    run_dir: Optional[str] = None,
) -> str:
    run_dir = run_dir or os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    for sub in ["contracts", "agents", "sandbox", "artifacts"]:
        _ensure_dir(os.path.join(run_dir, sub))
    register_run_log(run_id, os.path.join(run_dir, "events.jsonl"))
    _RUN_DIRS[run_id] = run_dir
    _RUN_ATTEMPTS.setdefault(run_id, [])

    if enable_tee:
        app_log_path = os.path.join(run_dir, "app_log.txt")
        _ensure_dir(os.path.dirname(app_log_path))
        try:
            log_handle = open(app_log_path, "a", encoding="utf-8")
            global _TEE_STREAM
            if _TEE_STREAM is None:
                _TEE_STREAM = _TeeStream(sys.stdout, log_handle)
                sys.stdout = _TEE_STREAM
                sys.stderr = _TEE_STREAM
        except Exception:
            pass
    if state is not None:
        state["run_bundle_dir"] = run_dir
    return run_dir


def get_run_dir(run_id: str) -> Optional[str]:
    return _RUN_DIRS.get(run_id)


def log_agent_snapshot(
    run_id: str,
    agent: str,
    prompt: Optional[str] = None,
    response: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    script: Optional[str] = None,
    verdicts: Optional[Any] = None,
    attempt: Optional[int] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    base = os.path.join(run_dir, "agents", agent)
    if attempt is not None:
        base = os.path.join(base, f"attempt_{attempt}")
    _ensure_dir(base)
    if prompt:
        _write_text(os.path.join(base, "prompt.txt"), str(prompt))
    if response is not None:
        if isinstance(response, (dict, list)):
            _write_json(os.path.join(base, "response.json"), response)
        else:
            _write_text(os.path.join(base, "response.txt"), str(response))
    if context is not None:
        _write_json(os.path.join(base, "context.json"), context)
    if script:
        _write_text(os.path.join(base, "script.py"), script)
    if verdicts is not None:
        _write_json(os.path.join(base, "verdicts.json"), verdicts)


def log_sandbox_attempt(
    run_id: str,
    attempt: int,
    code: str,
    stdout: str,
    stderr: str,
    outputs_listing: Any,
    exit_code: Optional[int] = None,
    error_tail: Optional[str] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    attempt_dir = os.path.join(run_dir, "sandbox", f"attempt_{attempt}")
    _ensure_dir(attempt_dir)
    _write_text(os.path.join(attempt_dir, "code_sent.py"), code or "")
    _write_text(os.path.join(attempt_dir, "stdout.txt"), stdout or "")
    _write_text(os.path.join(attempt_dir, "stderr.txt"), stderr or "")
    if outputs_listing is not None:
        _write_json(os.path.join(attempt_dir, "outputs_listing.json"), outputs_listing)
    record = {
        "attempt": attempt,
        "exit_code": exit_code,
        "error_tail": error_tail,
    }
    _RUN_ATTEMPTS.setdefault(run_id, []).append(record)


def copy_run_artifacts(run_id: str, sources: List[str]) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    dest_root = os.path.join(run_dir, "artifacts")
    _ensure_dir(dest_root)
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        base_name = os.path.basename(src.rstrip("/\\"))
        dest = os.path.join(dest_root, base_name)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                _ensure_dir(dest_root)
                shutil.copy2(src, dest)
        except Exception:
            pass


def copy_run_contracts(run_id: str, sources: List[str]) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    dest_root = os.path.join(run_dir, "contracts")
    _ensure_dir(dest_root)
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        dest = os.path.join(dest_root, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
        except Exception:
            pass


def write_run_manifest(
    run_id: str,
    state: Dict[str, Any],
    status_final: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> Optional[str]:
    run_dir = get_run_dir(run_id) or os.path.join(RUNS_DIR, "latest")
    csv_path = state.get("csv_path") or ""
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract") or {}
    evaluation_spec = _safe_load_json("data/evaluation_spec.json") or state.get("evaluation_spec") or {}
    artifact_index = _safe_load_json("data/artifact_index.json") or state.get("artifact_index") or []
    output_contract = _safe_load_json("data/output_contract_report.json") or {}
    run_summary = _safe_load_json("data/run_summary.json") or {}
    required_outputs = _normalize_required_outputs(contract)
    produced_outputs = _normalize_produced_outputs(artifact_index)

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    existing = _safe_load_json(manifest_path)
    existing_dict = existing if isinstance(existing, dict) else {}

    gates_summary = {
        "status": run_summary.get("status") or state.get("review_verdict"),
        "failed_gates": run_summary.get("failed_gates", []) if isinstance(run_summary, dict) else [],
        "reason": (state.get("last_gate_context") or {}).get("feedback"),
    }

    manifest = dict(existing_dict)
    existing_input = existing_dict.get("input")
    if not csv_path and isinstance(existing_input, dict):
        csv_path = existing_input.get("path") or ""
    manifest.update(
        {
            "run_id": run_id,
            "started_at": started_at or existing_dict.get("started_at") or state.get("run_start_ts"),
            "ended_at": ended_at or datetime.utcnow().isoformat(),
            "git_commit": _git_commit(),
            "input": {
                "path": csv_path,
                "sha256": _hash_file(csv_path) or (existing_input or {}).get("sha256"),
                "dialect": {
                    "encoding": state.get("csv_encoding") or (existing_input or {}).get("dialect", {}).get("encoding"),
                    "sep": state.get("csv_sep") or (existing_input or {}).get("dialect", {}).get("sep"),
                    "decimal": state.get("csv_decimal") or (existing_input or {}).get("dialect", {}).get("decimal"),
                },
            },
            "models_by_agent": state.get("agent_models", {}) or existing_dict.get("models_by_agent", {}),
            "required_outputs": required_outputs,
            "produced_outputs": produced_outputs,
            "sandbox_attempts": _RUN_ATTEMPTS.get(run_id, []),
            "required_outputs_missing": output_contract.get("missing", []),
            "status_final": status_final or existing_dict.get("status_final") or gates_summary.get("status"),
            "gates_summary": gates_summary,
            "contracts": {
                "execution_contract": bool(contract),
                "evaluation_spec": bool(evaluation_spec),
                "artifact_index": bool(artifact_index),
                "contract_min": os.path.exists("data/contract_min.json"),
            },
        }
    )
    _write_json(manifest_path, manifest)
    return manifest_path
