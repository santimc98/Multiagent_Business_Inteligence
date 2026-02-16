"""
Graph step modules extracted from graph.py (seniority refactoring).

Each module contains step functions that were previously defined inline in graph.py.
The graph routing logic remains in graph.py; step execution is delegated here.
"""

from src.graph.steps.contract_resolution import (
    _resolve_contract_columns,
    _resolve_contract_columns_for_cleaning,
    _resolve_required_outputs,
    _resolve_expected_output_paths,
    _resolve_allowed_columns_for_gate,
    _resolve_allowed_patterns_for_gate,
    _resolve_optional_runtime_downloads,
)

from src.graph.steps.context_builders import (
    _norm_name,
    _build_required_sample_context,
    _infer_parsing_hints_from_sample_context,
    _build_signal_summary_context,
    _build_cleaned_data_summary_min,
)

from src.graph.steps.result_evaluator import (
    _apply_review_consistency_guard,
    _harmonize_review_packets_with_final_eval,
    _looks_blocking_retry_signal,
)
from src.graph.steps.retry_policy import (
    should_reselect_strategy_on_retry,
)
from src.graph.steps.handoff_utils import (
    extract_preflight_gate_failures,
    extract_preflight_gate_tail,
)

__all__ = [
    "_resolve_contract_columns",
    "_resolve_contract_columns_for_cleaning",
    "_resolve_required_outputs",
    "_resolve_expected_output_paths",
    "_resolve_allowed_columns_for_gate",
    "_resolve_allowed_patterns_for_gate",
    "_resolve_optional_runtime_downloads",
    "_norm_name",
    "_build_required_sample_context",
    "_infer_parsing_hints_from_sample_context",
    "_build_signal_summary_context",
    "_build_cleaned_data_summary_min",
    "_apply_review_consistency_guard",
    "_harmonize_review_packets_with_final_eval",
    "_looks_blocking_retry_signal",
    "should_reselect_strategy_on_retry",
    "extract_preflight_gate_failures",
    "extract_preflight_gate_tail",
]
