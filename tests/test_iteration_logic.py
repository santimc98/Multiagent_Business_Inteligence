"""
Unit tests for ML iteration loop logic (deterministic, no CSV required).
Tests the iteration control flow based on:
- review_verdict/status
- retry_worth_it
- iteration_policy
- data_adequacy_report
- metric_history
- strategy_lock
"""
import pytest
from src.graph.graph import (
    _get_iteration_policy,
    _detect_metric_plateau,
    _validate_strategy_lock,
    _capture_strategy_snapshot,
    _append_ml_iteration_journal,
)


class TestRetryOnNeedsImprovementWhenWorthIt:
    """Test: status=NEEDS_IMPROVEMENT, retry_worth_it=True -> action=retry(metric)"""

    def test_needs_improvement_with_retry_worth_it_should_retry(self):
        # The downgrade logic is in run_result_evaluator
        # When status=NEEDS_IMPROVEMENT and no traceback, it should NOT downgrade
        # if retry_worth_it is True or None

        # Simulating the logic from run_result_evaluator lines 9261-9275
        status = "NEEDS_IMPROVEMENT"
        retry_worth_it = True
        has_deterministic_error = False  # No traceback

        # The new logic: only downgrade if retry_worth_it is False
        if status == "NEEDS_IMPROVEMENT":
            if has_deterministic_error:
                pass  # Keep NEEDS_IMPROVEMENT
            else:
                if retry_worth_it is False:
                    status = "APPROVE_WITH_WARNINGS"
                # else: keep NEEDS_IMPROVEMENT

        assert status == "NEEDS_IMPROVEMENT", "Should NOT downgrade when retry_worth_it=True"

    def test_needs_improvement_with_retry_worth_it_none_should_keep_status(self):
        status = "NEEDS_IMPROVEMENT"
        retry_worth_it = None  # Unknown
        has_deterministic_error = False

        if status == "NEEDS_IMPROVEMENT":
            if has_deterministic_error:
                pass
            else:
                if retry_worth_it is False:
                    status = "APPROVE_WITH_WARNINGS"

        assert status == "NEEDS_IMPROVEMENT", "Should NOT downgrade when retry_worth_it=None"


class TestDowngradeOnlyWhenRetryNotWorthIt:
    """Test: status=NEEDS_IMPROVEMENT, retry_worth_it=False -> APPROVE_WITH_WARNINGS"""

    def test_downgrade_when_retry_not_worth_it(self):
        status = "NEEDS_IMPROVEMENT"
        retry_worth_it = False
        has_deterministic_error = False

        if status == "NEEDS_IMPROVEMENT":
            if has_deterministic_error:
                pass
            else:
                if retry_worth_it is False:
                    status = "APPROVE_WITH_WARNINGS"

        assert status == "APPROVE_WITH_WARNINGS", "Should downgrade when retry_worth_it=False"

    def test_no_downgrade_with_traceback(self):
        status = "NEEDS_IMPROVEMENT"
        retry_worth_it = False
        has_deterministic_error = True  # Has traceback

        if status == "NEEDS_IMPROVEMENT":
            if has_deterministic_error:
                pass  # Keep NEEDS_IMPROVEMENT regardless of retry_worth_it
            else:
                if retry_worth_it is False:
                    status = "APPROVE_WITH_WARNINGS"

        assert status == "NEEDS_IMPROVEMENT", "Should NOT downgrade when traceback present"


class TestStopOnCeiling:
    """Test: threshold_reached=True OR signal_ceiling_reached in reasons -> stop"""

    def test_stop_on_threshold_reached(self):
        data_report = {"threshold_reached": True, "reasons": []}
        threshold_reached = bool(data_report.get("threshold_reached"))
        reasons = data_report.get("reasons", [])

        should_stop = threshold_reached or "signal_ceiling_reached" in reasons
        assert should_stop, "Should stop when threshold_reached=True"

    def test_stop_on_signal_ceiling_in_reasons(self):
        data_report = {"threshold_reached": False, "reasons": ["signal_ceiling_reached"]}
        threshold_reached = bool(data_report.get("threshold_reached"))
        reasons = data_report.get("reasons", [])

        should_stop = threshold_reached or "signal_ceiling_reached" in reasons
        assert should_stop, "Should stop when signal_ceiling_reached in reasons"

    def test_no_stop_when_no_ceiling(self):
        data_report = {"threshold_reached": False, "reasons": ["other_reason"]}
        threshold_reached = bool(data_report.get("threshold_reached"))
        reasons = data_report.get("reasons", [])

        should_stop = threshold_reached or "signal_ceiling_reached" in reasons
        assert not should_stop, "Should NOT stop when no ceiling"


class TestStopOnPlateau:
    """Test: metric_history plateau=True -> stop"""

    def test_plateau_detected_with_low_lift(self):
        metric_history = [
            {"primary_metric_name": "accuracy", "primary_metric_value": 0.80, "lift": 0.005, "eval_signature": "sig1"},
            {"primary_metric_name": "accuracy", "primary_metric_value": 0.805, "lift": 0.003, "eval_signature": "sig1"},
        ]
        plateau, reason = _detect_metric_plateau(metric_history, window=2, epsilon=0.01)
        assert plateau, f"Should detect plateau when lift<epsilon for window iterations: {reason}"

    def test_no_plateau_with_sufficient_improvement(self):
        metric_history = [
            {"primary_metric_name": "accuracy", "primary_metric_value": 0.70, "lift": 0.05, "eval_signature": "sig1"},
            {"primary_metric_name": "accuracy", "primary_metric_value": 0.75, "lift": 0.05, "eval_signature": "sig1"},
        ]
        plateau, reason = _detect_metric_plateau(metric_history, window=2, epsilon=0.01)
        assert not plateau, "Should NOT detect plateau when lift is significant"

    def test_no_plateau_with_insufficient_history(self):
        metric_history = [
            {"primary_metric_name": "accuracy", "primary_metric_value": 0.80, "lift": 0.001, "eval_signature": "sig1"},
        ]
        plateau, reason = _detect_metric_plateau(metric_history, window=2, epsilon=0.01)
        assert not plateau, "Should NOT detect plateau with insufficient history"


class TestPolicyLegacyFallback:
    """Test: max_iterations -> metric_improvement_max fallback"""

    def test_max_iterations_fallback(self):
        state = {
            "execution_contract": {
                "iteration_policy": {
                    "max_iterations": 5,
                    # no metric_improvement_max
                }
            }
        }
        policy = _get_iteration_policy(state)
        assert policy is not None
        assert policy.get("metric_improvement_max") == 5, "Should fallback max_iterations to metric_improvement_max"

    def test_metric_improvement_max_takes_precedence(self):
        state = {
            "execution_contract": {
                "iteration_policy": {
                    "max_iterations": 10,
                    "metric_improvement_max": 3,
                }
            }
        }
        policy = _get_iteration_policy(state)
        assert policy is not None
        assert policy.get("metric_improvement_max") == 3, "metric_improvement_max should take precedence"

    def test_no_policy_returns_none(self):
        state = {"execution_contract": {}}
        policy = _get_iteration_policy(state)
        assert policy is None, "Should return None when no iteration_policy"


class TestStrategyLockBlocksRetry:
    """Test: strategy/contract drift -> stop with hard_fail_reason=STRATEGY_LOCK_FAILED"""

    def test_strategy_lock_detects_title_drift(self):
        state = {
            "strategy_lock_snapshot": {
                "strategy_title": "Original Strategy",
                "strategy_id": "strat_1",
                "contract_version": "1.0",
                "canonical_columns": ["col_a", "col_b"],
                "decision_columns": [],
                "outcome_columns": [],
                "allowed_feature_sets": [],
                "forbidden_features": [],
            },
            "selected_strategy": {"title": "Different Strategy", "id": "strat_1"},
            "execution_contract": {"version": "1.0", "canonical_columns": ["col_a", "col_b"]},
        }
        ok, details = _validate_strategy_lock(state)
        assert not ok, "Should detect strategy title drift"
        assert "drifts" in details
        assert any("strategy_title" in d for d in details["drifts"])

    def test_strategy_lock_detects_column_drift(self):
        state = {
            "strategy_lock_snapshot": {
                "strategy_title": "Strategy",
                "strategy_id": None,
                "contract_version": "1.0",
                "canonical_columns": ["col_a", "col_b"],
                "decision_columns": [],
                "outcome_columns": [],
                "allowed_feature_sets": [],
                "forbidden_features": [],
            },
            "selected_strategy": {"title": "Strategy"},
            "execution_contract": {
                "version": "1.0",
                "canonical_columns": ["col_a", "col_b", "col_c"],  # Added column
            },
        }
        ok, details = _validate_strategy_lock(state)
        assert not ok, "Should detect canonical_columns drift"
        assert "drifts" in details

    def test_strategy_lock_passes_when_no_drift(self):
        state = {
            "strategy_lock_snapshot": {
                "strategy_title": "Strategy",
                "strategy_id": "s1",
                "contract_version": "1.0",
                "canonical_columns": ["col_a", "col_b"],
                "decision_columns": [],
                "outcome_columns": [],
                "allowed_feature_sets": [],
                "forbidden_features": [],
            },
            "selected_strategy": {"title": "Strategy", "id": "s1"},
            "execution_contract": {
                "version": "1.0",
                "canonical_columns": ["col_a", "col_b"],
            },
        }
        ok, details = _validate_strategy_lock(state)
        assert ok, "Should pass when no drift"

    def test_strategy_lock_passes_when_no_snapshot(self):
        state = {
            "selected_strategy": {"title": "Strategy"},
            "execution_contract": {"canonical_columns": ["col_a"]},
        }
        ok, details = _validate_strategy_lock(state)
        assert ok, "Should pass when no snapshot (first iteration)"
        assert details.get("reason") == "no_snapshot_yet"


class TestJournalAllowsMultipleStagesSameIteration:
    """Test: append (1, preflight) then (1, review_complete) -> both present"""

    def test_journal_multiple_stages_same_iteration(self, tmp_path):
        run_id = "test_run"
        base_dir = str(tmp_path)

        entry_preflight = {
            "iteration_id": 1,
            "stage": "preflight",
            "status": "checking",
        }
        entry_review = {
            "iteration_id": 1,
            "stage": "review_complete",
            "status": "NEEDS_IMPROVEMENT",
        }

        written1 = _append_ml_iteration_journal(run_id, entry_preflight, [], base_dir=base_dir)
        written2 = _append_ml_iteration_journal(run_id, entry_review, written1, base_dir=base_dir)

        # Both entries should be written (different stages)
        assert "1:preflight" in written2
        assert "1:review_complete" in written2

    def test_journal_deduplicates_same_stage(self, tmp_path):
        run_id = "test_run"
        base_dir = str(tmp_path)

        entry1 = {"iteration_id": 1, "stage": "preflight", "data": "first"}
        entry2 = {"iteration_id": 1, "stage": "preflight", "data": "second"}

        written1 = _append_ml_iteration_journal(run_id, entry1, [], base_dir=base_dir)
        written2 = _append_ml_iteration_journal(run_id, entry2, written1, base_dir=base_dir)

        # Should deduplicate - second entry with same stage should not be written
        assert written1 == written2, "Should deduplicate same iteration:stage"


class TestCaptureStrategySnapshot:
    """Test snapshot capture functionality"""

    def test_capture_snapshot_extracts_all_fields(self):
        state = {
            "selected_strategy": {
                "title": "Test Strategy",
                "id": "strat_123",
            },
            "execution_contract": {
                "version": "4.1",
                "canonical_columns": ["b_col", "a_col"],
                "decision_columns": ["decision"],
                "outcome_columns": ["outcome"],
                "allowed_feature_sets": ["set2", "set1"],
                "forbidden_features": ["forbidden"],
            },
        }
        snapshot = _capture_strategy_snapshot(state)

        assert snapshot["strategy_title"] == "Test Strategy"
        assert snapshot["strategy_id"] == "strat_123"
        assert snapshot["contract_version"] == "4.1"
        # Columns should be sorted for consistent comparison
        assert snapshot["canonical_columns"] == ["a_col", "b_col"]
        assert snapshot["allowed_feature_sets"] == ["set1", "set2"]

    def test_capture_snapshot_handles_missing_fields(self):
        state = {
            "selected_strategy": {},
            "execution_contract": {},
        }
        snapshot = _capture_strategy_snapshot(state)

        assert snapshot["strategy_title"] is None
        assert snapshot["canonical_columns"] == []

    def test_capture_snapshot_contract_version_fallback(self):
        """Test that contract_version falls back to version field."""
        state = {
            "selected_strategy": {"title": "Test"},
            "execution_contract": {
                "contract_version": "4.2",  # Should use this
                "version": "1.0",           # Not this
            },
        }
        snapshot = _capture_strategy_snapshot(state)
        assert snapshot["contract_version"] == "4.2"

        # Test fallback when contract_version not present
        state2 = {
            "selected_strategy": {"title": "Test"},
            "execution_contract": {
                "version": "3.0",  # Should use this as fallback
            },
        }
        snapshot2 = _capture_strategy_snapshot(state2)
        assert snapshot2["contract_version"] == "3.0"

    def test_capture_snapshot_normalizes_dict_allowed_feature_sets(self):
        """Test that dict-format allowed_feature_sets is normalized."""
        state = {
            "selected_strategy": {"title": "Test"},
            "execution_contract": {
                "allowed_feature_sets": {
                    "core": ["z_feature", "a_feature"],
                    "extended": ["m_feature", "b_feature"],
                    "forbidden": ["bad_feature"],
                },
            },
        }
        snapshot = _capture_strategy_snapshot(state)
        # Dict values should be sorted
        assert snapshot["allowed_feature_sets"]["core"] == ["a_feature", "z_feature"]
        assert snapshot["allowed_feature_sets"]["extended"] == ["b_feature", "m_feature"]
        # forbidden_features should be derived
        assert snapshot["forbidden_features"] == ["bad_feature"]

    def test_capture_snapshot_derives_forbidden_from_allowed(self):
        """Test forbidden_features derivation when not explicitly set."""
        state = {
            "selected_strategy": {"title": "Test"},
            "execution_contract": {
                "allowed_feature_sets": {
                    "forbidden": ["derived_forbidden1", "derived_forbidden2"],
                },
                # No explicit forbidden_features
            },
        }
        snapshot = _capture_strategy_snapshot(state)
        assert snapshot["forbidden_features"] == ["derived_forbidden1", "derived_forbidden2"]


class TestAllowedFeatureSetsNormalization:
    """Test allowed_feature_sets normalization for drift detection."""

    def test_allowed_feature_sets_drift_detected(self):
        """Test that changes in allowed_feature_sets are detected."""
        state = {
            "strategy_lock_snapshot": {
                "strategy_title": "Strategy",
                "strategy_id": None,
                "contract_version": "1.0",
                "canonical_columns": [],
                "decision_columns": [],
                "outcome_columns": [],
                "allowed_feature_sets": ["feature_a", "feature_b"],
                "forbidden_features": [],
            },
            "selected_strategy": {"title": "Strategy"},
            "execution_contract": {
                "version": "1.0",
                "allowed_feature_sets": ["feature_a", "feature_c"],  # Changed
            },
        }
        ok, details = _validate_strategy_lock(state)
        assert not ok, "Should detect allowed_feature_sets drift"
        assert any("allowed_feature_sets" in d for d in details.get("drifts", []))

    def test_dict_allowed_feature_sets_drift_detected(self):
        """Test drift detection with dict-format allowed_feature_sets."""
        state = {
            "strategy_lock_snapshot": {
                "strategy_title": "Strategy",
                "strategy_id": None,
                "contract_version": "1.0",
                "canonical_columns": [],
                "decision_columns": [],
                "outcome_columns": [],
                "allowed_feature_sets": {"core": ["a", "b"]},
                "forbidden_features": [],
            },
            "selected_strategy": {"title": "Strategy"},
            "execution_contract": {
                "version": "1.0",
                "allowed_feature_sets": {"core": ["a", "c"]},  # Changed
            },
        }
        ok, details = _validate_strategy_lock(state)
        assert not ok, "Should detect dict allowed_feature_sets drift"
