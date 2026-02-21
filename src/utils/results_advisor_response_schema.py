"""
Structured-output schema for ResultsAdvisor critique packets.

This schema is intentionally strict on top-level structure while keeping
validation_signals flexible enough for cv/holdout variants.
"""

from __future__ import annotations

from typing import Any, Dict


def build_results_advisor_critique_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "packet_type",
            "packet_version",
            "run_id",
            "iteration",
            "timestamp_utc",
            "primary_metric_name",
            "higher_is_better",
            "metric_comparison",
            "validation_signals",
            "error_modes",
            "risk_flags",
            "active_gates_context",
            "analysis_summary",
            "strictly_no_code_advice",
        ],
        "properties": {
            "packet_type": {"type": "string", "enum": ["advisor_critique_packet"]},
            "packet_version": {"type": "string", "enum": ["1.0"]},
            "run_id": {"type": "string"},
            "iteration": {"type": "integer", "minimum": 0},
            "timestamp_utc": {"type": "string"},
            "primary_metric_name": {"type": "string"},
            "higher_is_better": {"type": "boolean"},
            "metric_comparison": {
                "type": "object",
                "required": [
                    "baseline_value",
                    "candidate_value",
                    "delta_abs",
                    "delta_rel",
                    "min_delta_required",
                    "meets_min_delta",
                ],
                "properties": {
                    "baseline_value": {"type": "number"},
                    "candidate_value": {"type": "number"},
                    "delta_abs": {"type": "number"},
                    "delta_rel": {"type": "number"},
                    "min_delta_required": {"type": "number"},
                    "meets_min_delta": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "validation_signals": {
                "type": "object",
                "required": ["validation_mode"],
                "properties": {
                    "validation_mode": {
                        "type": "string",
                        "enum": ["cv", "holdout", "cv_and_holdout", "unknown"],
                    },
                    "cv": {
                        "type": "object",
                        "required": ["cv_mean", "cv_std", "fold_count", "variance_level"],
                        "properties": {
                            "cv_mean": {"type": "number"},
                            "cv_std": {"type": "number"},
                            "fold_count": {"type": "integer", "minimum": 2},
                            "variance_level": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "unknown"],
                            },
                        },
                        "additionalProperties": False,
                    },
                    "holdout": {
                        "type": "object",
                        "required": [
                            "metric_value",
                            "split_name",
                            "sample_count",
                            "class_distribution_shift",
                        ],
                        "properties": {
                            "metric_value": {"type": "number"},
                            "split_name": {"type": "string"},
                            "sample_count": {"type": "integer", "minimum": 1},
                            "class_distribution_shift": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "unknown"],
                            },
                            "positive_class_rate": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                    "generalization_gap": {"type": "number"},
                },
                "additionalProperties": False,
            },
            "error_modes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "id",
                        "severity",
                        "confidence",
                        "evidence",
                        "affected_scope",
                        "metric_impact_direction",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                        "affected_scope": {"type": "string"},
                        "metric_impact_direction": {
                            "type": "string",
                            "enum": ["negative", "neutral", "positive"],
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "active_gates_context": {"type": "array", "items": {"type": "string"}},
            "analysis_summary": {"type": "string"},
            "strictly_no_code_advice": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

