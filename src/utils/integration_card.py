from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.utils.contract_accessors import get_declared_artifacts


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _norm_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _artifact_exists(path: str, work_dir: str = ".") -> bool:
    if not path:
        return False
    candidates = [path]
    if work_dir and work_dir not in {".", ""}:
        candidates.append(os.path.join(work_dir, path))
    return any(os.path.exists(candidate) for candidate in candidates)


def _extract_schema_columns(schema: Any) -> List[Dict[str, Any]]:
    columns: List[Dict[str, Any]] = []
    if isinstance(schema, dict):
        raw_cols = (
            schema.get("columns")
            or schema.get("required_columns")
            or schema.get("output_columns")
            or schema.get("fields")
        )
        if isinstance(raw_cols, list):
            for item in raw_cols[:200]:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("column") or item.get("field")
                    if name:
                        columns.append(
                            {
                                "name": str(name),
                                "type": str(item.get("type") or item.get("dtype") or ""),
                                "description": str(item.get("description") or item.get("meaning") or ""),
                                "required": bool(item.get("required", True)),
                            }
                        )
                elif isinstance(item, str):
                    columns.append({"name": item, "type": "", "description": "", "required": True})
        props = schema.get("properties")
        if isinstance(props, dict):
            for name, spec in list(props.items())[:200]:
                spec = spec if isinstance(spec, dict) else {}
                columns.append(
                    {
                        "name": str(name),
                        "type": str(spec.get("type") or spec.get("dtype") or ""),
                        "description": str(spec.get("description") or ""),
                        "required": True,
                    }
                )
    elif isinstance(schema, list):
        for item in schema[:200]:
            if isinstance(item, str):
                columns.append({"name": item, "type": "", "description": "", "required": True})
            elif isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("field")
                if name:
                    columns.append(
                        {
                            "name": str(name),
                            "type": str(item.get("type") or item.get("dtype") or ""),
                            "description": str(item.get("description") or ""),
                            "required": bool(item.get("required", True)),
                        }
                    )
    seen = set()
    unique = []
    for col in columns:
        key = str(col.get("name") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(col)
    return unique


def _schemas_by_path(contract: Dict[str, Any]) -> Dict[str, Any]:
    schemas: Dict[str, Any] = {}
    artifact_reqs = contract.get("artifact_requirements") if isinstance(contract.get("artifact_requirements"), dict) else {}
    file_schemas = artifact_reqs.get("file_schemas") if isinstance(artifact_reqs.get("file_schemas"), dict) else {}
    for path, schema in file_schemas.items():
        schemas[_norm_path(path).lower()] = schema
    for key in ("scored_rows_schema", "schema_binding", "output_schema"):
        schema = artifact_reqs.get(key)
        if isinstance(schema, dict):
            target = schema.get("path") or schema.get("artifact") or schema.get("output_path")
            if target:
                schemas[_norm_path(target).lower()] = schema
    artifact_schemas = contract.get("artifact_schemas")
    if isinstance(artifact_schemas, dict):
        for path, schema in artifact_schemas.items():
            schemas[_norm_path(path).lower()] = schema
    return schemas


def _extract_input_columns(
    contract: Dict[str, Any],
    model_card: Dict[str, Any],
    metrics_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Any] = []
    for payload in (model_card, metrics_payload):
        for key in ("features_used", "feature_names", "model_features", "features", "input_features"):
            raw = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(raw, list) and raw:
                candidates = raw
                break
        if candidates:
            break
    if not candidates:
        for key in ("canonical_columns", "model_input_candidates", "required_columns"):
            raw = contract.get(key)
            if isinstance(raw, list) and raw:
                candidates = raw
                break
    columns = []
    for item in candidates[:300]:
        if isinstance(item, dict):
            name = item.get("name") or item.get("column") or item.get("feature")
            if name:
                columns.append(
                    {
                        "name": str(name),
                        "type": str(item.get("type") or item.get("dtype") or ""),
                        "description": str(item.get("description") or item.get("meaning") or ""),
                        "required": bool(item.get("required", True)),
                    }
                )
        elif isinstance(item, str):
            columns.append({"name": item, "type": "", "description": "", "required": True})
    return columns


def _find_entrypoints(declared_artifacts: List[Dict[str, Any]], work_dir: str) -> List[Dict[str, Any]]:
    candidates = []
    known = ["scoring_function.py", "src/scoring_function.py", "artifacts/ml/scoring_function.py"]
    for artifact in declared_artifacts:
        path = _norm_path((artifact or {}).get("path"))
        lower = path.lower()
        if lower.endswith(".py") and ("scor" in lower or "predict" in lower or "infer" in lower):
            candidates.append(path)
    candidates.extend(known)
    seen = set()
    out = []
    for path in candidates:
        norm = _norm_path(path)
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "path": norm,
                "exists": _artifact_exists(norm, work_dir),
                "expected_interface": "predict(df_features) -> dataframe with prediction columns when provided by the ML artifact",
            }
        )
    return out[:12]


def build_integration_card(
    *,
    contract: Dict[str, Any] | None = None,
    model_card: Dict[str, Any] | None = None,
    metrics_payload: Dict[str, Any] | None = None,
    inference_benchmark: Dict[str, Any] | None = None,
    artifact_index: List[Dict[str, Any]] | None = None,
    model_dependency_context_pack: Dict[str, Any] | None = None,
    work_dir: str = ".",
) -> Dict[str, Any]:
    contract = contract if isinstance(contract, dict) else {}
    model_card = model_card if isinstance(model_card, dict) else {}
    metrics_payload = metrics_payload if isinstance(metrics_payload, dict) else {}
    inference_benchmark = inference_benchmark if isinstance(inference_benchmark, dict) else {}
    artifact_index = artifact_index if isinstance(artifact_index, list) else []
    model_dependency_context_pack = (
        model_dependency_context_pack if isinstance(model_dependency_context_pack, dict) else {}
    )

    declared_artifacts = get_declared_artifacts(contract)
    schema_map = _schemas_by_path(contract)
    outputs = []
    for artifact in declared_artifacts:
        if not isinstance(artifact, dict):
            continue
        path = _norm_path(artifact.get("path"))
        if not path:
            continue
        schema = schema_map.get(path.lower())
        outputs.append(
            {
                "path": path,
                "intent": str(artifact.get("intent") or ""),
                "owner": str(artifact.get("owner") or ""),
                "kind": str(artifact.get("kind") or ""),
                "required": bool(artifact.get("required")),
                "exists": _artifact_exists(path, work_dir),
                "schema_columns": _extract_schema_columns(schema),
                "description": str(artifact.get("description") or ""),
            }
        )

    if artifact_index:
        known_paths = {str(item.get("path") or "").replace("\\", "/").lower() for item in outputs}
        for item in artifact_index:
            if not isinstance(item, dict):
                continue
            path = _norm_path(item.get("path"))
            if not path or path.lower() in known_paths:
                continue
            outputs.append(
                {
                    "path": path,
                    "intent": str(item.get("type") or item.get("intent") or ""),
                    "owner": "",
                    "kind": str(item.get("kind") or ""),
                    "required": False,
                    "exists": bool(item.get("present", True)),
                    "schema_columns": [],
                    "description": "Produced artifact discovered in artifact index.",
                }
            )

    dependency_signals = (
        model_dependency_context_pack.get("model_dependency_signals")
        if isinstance(model_dependency_context_pack.get("model_dependency_signals"), dict)
        else {}
    )
    runtime_payload = inference_benchmark or model_card.get("inference_benchmark") or metrics_payload.get("inference_benchmark") or {}
    model_size_bytes = (
        _safe_int(model_card.get("model_size_bytes"))
        or _safe_int(metrics_payload.get("model_size_bytes"))
        or _safe_int((runtime_payload or {}).get("model_size_bytes") if isinstance(runtime_payload, dict) else None)
    )
    return {
        "schema_version": "1.0",
        "role": "integration_handoff_advisory",
        "deterministic_policy": (
            "This card organizes integration facts and missing handoff evidence. It must not reject or approve "
            "the model by itself; agents use it to reason and report."
        ),
        "business_context": {
            "objective": str(contract.get("business_objective") or ""),
            "decisioning_requirements": contract.get("decisioning_requirements") if isinstance(contract.get("decisioning_requirements"), dict) else {},
        },
        "input_contract": {
            "required_input_columns": _extract_input_columns(contract, model_card, metrics_payload),
            "feature_count": len(_extract_input_columns(contract, model_card, metrics_payload)),
            "source": "model_card_or_metrics_or_contract",
        },
        "execution_contract": {
            "entrypoints": _find_entrypoints(declared_artifacts, work_dir),
            "model_artifact_paths": [
                item.get("path")
                for item in outputs
                if str(item.get("path") or "").lower().endswith((".pkl", ".joblib", ".pickle", ".onnx"))
            ],
            "runtime_requirements": {
                "inference_benchmark": runtime_payload if isinstance(runtime_payload, dict) else {},
                "model_size_bytes": model_size_bytes,
            },
        },
        "output_contract": {
            "artifacts": outputs[:120],
            "required_artifact_count": sum(1 for item in outputs if item.get("required")),
            "missing_required_artifacts": [
                item.get("path") for item in outputs if item.get("required") and not item.get("exists")
            ],
        },
        "operational_dependency_context": {
            "top_features": dependency_signals.get("top_features", [])[:15] if isinstance(dependency_signals, dict) else [],
            "source_family_shares": dependency_signals.get("source_family_shares", [])[:12] if isinstance(dependency_signals, dict) else [],
            "top_feature_share": dependency_signals.get("top_feature_share") if isinstance(dependency_signals, dict) else None,
            "top_source_family_share": dependency_signals.get("top_source_family_share") if isinstance(dependency_signals, dict) else None,
        },
        "senior_integration_questions": [
            "Can an engineering team call the scoring entrypoint without reading the training script?",
            "Are required input columns, prediction columns, probability columns, and artifact paths explicit enough for production wiring?",
            "Are latency, model size, memory, and dependency assumptions measured or still missing?",
            "Does the report separate model performance from integration readiness and operational caveats?",
        ],
    }


def summarize_integration_card(card: Dict[str, Any] | None, *, max_lines: int = 80) -> str:
    if not isinstance(card, dict) or not card:
        return ""
    input_contract = card.get("input_contract") if isinstance(card.get("input_contract"), dict) else {}
    execution_contract = card.get("execution_contract") if isinstance(card.get("execution_contract"), dict) else {}
    output_contract = card.get("output_contract") if isinstance(card.get("output_contract"), dict) else {}
    op_dep = card.get("operational_dependency_context") if isinstance(card.get("operational_dependency_context"), dict) else {}
    entrypoints = execution_contract.get("entrypoints") if isinstance(execution_contract.get("entrypoints"), list) else []
    artifacts = output_contract.get("artifacts") if isinstance(output_contract.get("artifacts"), list) else []
    lines = [
        "INTEGRATION_CARD_SUMMARY:",
        "- role: integration_handoff_advisory; production handoff facts, not deterministic approval/rejection",
        f"- feature_count: {input_contract.get('feature_count')}; required_artifact_count: {output_contract.get('required_artifact_count')}; missing_required_artifacts: {output_contract.get('missing_required_artifacts')}",
        f"- entrypoints: {entrypoints[:6]}",
        f"- model_artifact_paths: {execution_contract.get('model_artifact_paths')}",
        f"- runtime_requirements: {execution_contract.get('runtime_requirements')}",
        f"- operational_dependency: top_feature_share={op_dep.get('top_feature_share')}, top_source_family_share={op_dep.get('top_source_family_share')}",
    ]
    if artifacts:
        lines.append("- artifacts_sample:")
        for artifact in artifacts[:10]:
            if isinstance(artifact, dict):
                lines.append(
                    "  "
                    + str(
                        {
                            "path": artifact.get("path"),
                            "intent": artifact.get("intent"),
                            "required": artifact.get("required"),
                            "exists": artifact.get("exists"),
                            "schema_columns": artifact.get("schema_columns", [])[:8],
                        }
                    )
                )
    questions = card.get("senior_integration_questions") if isinstance(card.get("senior_integration_questions"), list) else []
    if questions:
        lines.append("- senior_integration_questions:")
        for question in questions[:8]:
            lines.append(f"  - {question}")
    return "\n".join(lines[: max(1, int(max_lines))])
