from src.graph.graph import (
    _is_improvement,
    _metric_round_stability_ok,
    _resolve_contract_metric_target,
    _resolve_contract_primary_metric_name,
)


def test_resolve_contract_primary_metric_name_prefers_competition_metric() -> None:
    contract = {
        "validation_requirements": {
            "primary_metric": "roc_auc",
            "competition_metric": "normalized_gini",
        }
    }
    assert _resolve_contract_primary_metric_name({}, contract) == "normalized_gini"


def test_resolve_contract_metric_target_reads_explicit_direction() -> None:
    contract = {
        "validation_requirements": {
            "competition_metric": {
                "name": "rmse",
                "higher_is_better": False,
            }
        }
    }
    target = _resolve_contract_metric_target({}, contract)
    assert target.get("name") == "rmse"
    assert target.get("higher_is_better") is False


def test_metric_round_stability_ok_rejects_instability_error_modes() -> None:
    critique_packet = {
        "error_modes": [{"id": "fold_instability"}],
        "validation_signals": {"validation_mode": "cv"},
    }
    assert _metric_round_stability_ok(critique_packet) is False


def test_metric_round_stability_ok_rejects_large_generalization_gap() -> None:
    critique_packet = {
        "error_modes": [],
        "validation_signals": {
            "validation_mode": "cv_and_holdout",
            "generalization_gap": 0.031,
            "cv": {"cv_std": 0.01, "variance_level": "low"},
        },
    }
    assert _metric_round_stability_ok(critique_packet, max_generalization_gap=0.02) is False


def test_is_improvement_requires_stability_gate() -> None:
    assert _is_improvement(0.8, 0.801, True, 0.0005, stability_ok=False) is False
    assert _is_improvement(0.8, 0.801, True, 0.0005, stability_ok=True) is True
