from src.graph.graph import (
    _merge_non_empty_policy,
    _ensure_plot_spec_in_policy,
    _maybe_set_contract_min_policy,
)


def test_merge_policy_preserves_plot_spec_and_constraints():
    derived = {"sections": ["Executive"], "constraints": {"no_markdown_tables": True}}
    existing = {
        "plot_spec": {"enabled": True, "plots": [{"id": "plot_1"}], "max_plots": 1},
        "constraints": {"language": "es"},
    }
    merged = _merge_non_empty_policy(derived, existing)
    merged = _ensure_plot_spec_in_policy(merged, {"strategy_title": "Test"})
    assert merged.get("plot_spec", {}).get("plots")
    assert merged.get("constraints", {}).get("no_markdown_tables") is True
    assert merged.get("constraints", {}).get("language") == "es"


def test_contract_min_policy_not_overwritten_when_present():
    contract_min = {"reporting_policy": {"plot_spec": {"enabled": False, "max_plots": 2}}}
    policy = {"plot_spec": {"enabled": True, "plots": [{"id": "plot_1"}], "max_plots": 1}}
    _maybe_set_contract_min_policy(contract_min, policy)
    assert contract_min["reporting_policy"]["plot_spec"]["enabled"] is False


def test_contract_min_policy_compact_fallback():
    contract_min = {}
    policy = {"plot_spec": {"enabled": True, "plots": [{"id": "plot_1"}, {"id": "plot_2"}], "max_plots": 3}}
    _maybe_set_contract_min_policy(contract_min, policy)
    compact = contract_min.get("reporting_policy", {}).get("plot_spec", {})
    assert set(compact.keys()) == {"enabled", "max_plots"}
    assert compact.get("enabled") is True
    assert compact.get("max_plots") == 3
