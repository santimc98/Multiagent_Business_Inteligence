from src.graph import graph as graph_mod


def _profile(n_rows: int, n_cols: int, dtype: str = "float64"):
    return {
        "basic_stats": {
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
        "dtypes": {f"c{i}": dtype for i in range(n_cols)},
    }


def test_backend_selection_prefers_e2b_when_memory_is_safe():
    state = {
        "dataset_scale_hints": {
            "file_mb": 15.0,
            "est_rows": 120_000,
            "cols": 30,
        }
    }
    data_profile = _profile(120_000, 30, "float64")
    ml_plan = {"cv_policy": {"n_splits": 5}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is False
    assert reason == "e2b_memory_safe"
    decision = state.get("backend_memory_estimate") or {}
    assert decision.get("estimate", {}).get("estimated_peak_gb") is not None


def test_backend_selection_uses_heavy_when_memory_exceeds_safe_budget():
    state = {
        "dataset_scale_hints": {
            "file_mb": 350.0,
            "est_rows": 2_500_000,
            "cols": 200,
        }
    }
    data_profile = _profile(2_500_000, 200, "float64")
    ml_plan = {"cv_policy": {"n_splits": 5}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is True
    assert reason == "est_memory_exceeds_e2b_safe"
    decision = state.get("backend_memory_estimate") or {}
    estimate = decision.get("estimate") or {}
    assert (estimate.get("estimated_peak_gb") or 0) > (decision.get("safe_budget_gb") or 0)


def test_backend_selection_promotes_to_heavy_after_e2b_memory_failure():
    state = {
        "e2b_memory_failure_detected": True,
    }
    data_profile = _profile(50_000, 20, "float64")
    ml_plan = {"cv_policy": {"n_splits": 3}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is True
    assert reason == "e2b_memory_failure_fallback"


def test_memory_pressure_detector_matches_common_oom_signatures():
    assert graph_mod._is_memory_pressure_error("MemoryError: Unable to allocate 3.2 GiB")
    assert graph_mod._is_memory_pressure_error("Process Killed (exit code 137)")
    assert not graph_mod._is_memory_pressure_error("ValueError: invalid literal for int()")
