"""
Execution Planner structured-output schemas.
"""

from typing import Any, Dict


EXECUTION_CONTRACT_V41_MIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "contract_version",
        "scope",
        "strategy_title",
        "business_objective",
        "output_dialect",
        "canonical_columns",
        "column_roles",
        "allowed_feature_sets",
        "artifact_requirements",
        "required_outputs",
        "iteration_policy",
    ],
    "properties": {
        "contract_version": {"type": "string"},
        "scope": {
            "type": "string",
            "enum": ["cleaning_only", "ml_only", "full_pipeline"],
        },
        "strategy_title": {"type": "string"},
        "business_objective": {"type": "string"},
        "output_dialect": {
            "type": "object",
            "required": ["sep", "decimal", "encoding"],
            "properties": {
                "sep": {"type": "string"},
                "decimal": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "canonical_columns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "required_outputs": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "column_roles": {
            "type": "object",
            "required": [
                "pre_decision",
                "decision",
                "outcome",
                "post_decision_audit_only",
                "unknown",
                "identifiers",
                "time_columns",
            ],
            "properties": {
                "pre_decision": {"type": "array", "items": {"type": "string"}},
                "decision": {"type": "array", "items": {"type": "string"}},
                "outcome": {"type": "array", "items": {"type": "string"}},
                "post_decision_audit_only": {"type": "array", "items": {"type": "string"}},
                "unknown": {"type": "array", "items": {"type": "string"}},
                "identifiers": {"type": "array", "items": {"type": "string"}},
                "time_columns": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "allowed_feature_sets": {
            "type": "object",
            "required": [
                "segmentation_features",
                "model_features",
                "forbidden_features",
                "audit_only_features",
            ],
            "properties": {
                "segmentation_features": {"type": "array", "items": {"type": "string"}},
                "model_features": {"type": "array", "items": {"type": "string"}},
                "forbidden_features": {"type": "array", "items": {"type": "string"}},
                "audit_only_features": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "artifact_requirements": {
            "type": "object",
            "additionalProperties": True,
        },
        "iteration_policy": {
            "type": "object",
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

