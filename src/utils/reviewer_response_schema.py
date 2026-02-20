"""
Structured-output schemas for Reviewer/QA reviewer Gemini responses.

If pydantic is available, schemas are derived from strict models.
Otherwise, deterministic JSON Schema fallbacks are used.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

try:
    from pydantic import BaseModel, ConfigDict, Field

    _HAS_PYDANTIC = True
except Exception:
    BaseModel = object  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _HAS_PYDANTIC = False


def _normalize_gate_names(active_gate_names: List[str] | None) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for item in active_gate_names or []:
        gate = str(item or "").strip()
        if not gate:
            continue
        key = gate.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(gate)
    return normalized


def _inject_gate_enum(schema: Dict[str, Any], active_gate_names: List[str] | None) -> Dict[str, Any]:
    patched = copy.deepcopy(schema)
    gates = _normalize_gate_names(active_gate_names)
    if not gates:
        return patched
    properties = patched.get("properties")
    if not isinstance(properties, dict):
        return patched
    for key in ("failed_gates", "hard_failures"):
        gate_prop = properties.get(key)
        if isinstance(gate_prop, dict):
            gate_prop["items"] = {"type": "string", "enum": gates}
    return patched


if _HAS_PYDANTIC:
    class _EvidenceItem(BaseModel):
        claim: str = Field(min_length=1, max_length=500)
        source: str = Field(min_length=1, max_length=300)
        model_config = ConfigDict(extra="forbid")


    class _ImprovementSuggestions(BaseModel):
        techniques: List[str] = Field(default_factory=list)
        no_further_improvement: bool = False
        model_config = ConfigDict(extra="forbid")


    class _ReviewerDecision(BaseModel):
        status: str = Field(pattern="^(APPROVED|APPROVE_WITH_WARNINGS|REJECTED)$")
        feedback: str
        failed_gates: List[str] = Field(default_factory=list)
        required_fixes: List[str] = Field(default_factory=list)
        hard_failures: List[str] = Field(default_factory=list)
        evidence: List[_EvidenceItem] = Field(default_factory=list)
        improvement_suggestions: _ImprovementSuggestions = Field(
            default_factory=_ImprovementSuggestions
        )
        model_config = ConfigDict(extra="forbid")


    class _ReviewerEvalDecision(BaseModel):
        status: str = Field(pattern="^(APPROVED|NEEDS_IMPROVEMENT)$")
        feedback: str
        failed_gates: List[str] = Field(default_factory=list)
        required_fixes: List[str] = Field(default_factory=list)
        retry_worth_it: bool = True
        hard_failures: List[str] = Field(default_factory=list)
        evidence: List[_EvidenceItem] = Field(default_factory=list)
        model_config = ConfigDict(extra="forbid")


    class _QADecision(BaseModel):
        status: str = Field(pattern="^(APPROVED|APPROVE_WITH_WARNINGS|REJECTED)$")
        feedback: str
        failed_gates: List[str] = Field(default_factory=list)
        required_fixes: List[str] = Field(default_factory=list)
        hard_failures: List[str] = Field(default_factory=list)
        evidence: List[_EvidenceItem] = Field(default_factory=list)
        model_config = ConfigDict(extra="forbid")


def _fallback_reviewer_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "status",
            "feedback",
            "failed_gates",
            "required_fixes",
            "evidence",
            "improvement_suggestions",
        ],
        "properties": {
            "status": {"type": "string", "enum": ["APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"]},
            "feedback": {"type": "string"},
            "failed_gates": {"type": "array", "items": {"type": "string"}},
            "required_fixes": {"type": "array", "items": {"type": "string"}},
            "hard_failures": {"type": "array", "items": {"type": "string"}},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim", "source"],
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "improvement_suggestions": {
                "type": "object",
                "required": ["techniques", "no_further_improvement"],
                "properties": {
                    "techniques": {"type": "array", "items": {"type": "string"}},
                    "no_further_improvement": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def _fallback_reviewer_eval_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["status", "feedback", "failed_gates", "required_fixes", "retry_worth_it", "evidence"],
        "properties": {
            "status": {"type": "string", "enum": ["APPROVED", "NEEDS_IMPROVEMENT"]},
            "feedback": {"type": "string"},
            "failed_gates": {"type": "array", "items": {"type": "string"}},
            "required_fixes": {"type": "array", "items": {"type": "string"}},
            "retry_worth_it": {"type": "boolean"},
            "hard_failures": {"type": "array", "items": {"type": "string"}},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim", "source"],
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _fallback_qa_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["status", "feedback", "failed_gates", "required_fixes", "evidence"],
        "properties": {
            "status": {"type": "string", "enum": ["APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"]},
            "feedback": {"type": "string"},
            "failed_gates": {"type": "array", "items": {"type": "string"}},
            "required_fixes": {"type": "array", "items": {"type": "string"}},
            "hard_failures": {"type": "array", "items": {"type": "string"}},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim", "source"],
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def build_reviewer_response_schema(active_gate_names: List[str] | None = None) -> Dict[str, Any]:
    base = (
        _ReviewerDecision.model_json_schema()  # type: ignore[attr-defined]
        if _HAS_PYDANTIC
        else _fallback_reviewer_schema()
    )
    return _inject_gate_enum(base, active_gate_names)


def build_reviewer_eval_response_schema(active_gate_names: List[str] | None = None) -> Dict[str, Any]:
    base = (
        _ReviewerEvalDecision.model_json_schema()  # type: ignore[attr-defined]
        if _HAS_PYDANTIC
        else _fallback_reviewer_eval_schema()
    )
    return _inject_gate_enum(base, active_gate_names)


def build_qa_response_schema(active_gate_names: List[str] | None = None) -> Dict[str, Any]:
    base = (
        _QADecision.model_json_schema()  # type: ignore[attr-defined]
        if _HAS_PYDANTIC
        else _fallback_qa_schema()
    )
    return _inject_gate_enum(base, active_gate_names)

