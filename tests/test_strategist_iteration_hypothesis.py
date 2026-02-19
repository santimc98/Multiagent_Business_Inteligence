from src.agents.strategist import StrategistAgent
from src.utils.actor_critic_schemas import validate_iteration_hypothesis_packet


def test_generate_iteration_hypothesis_supports_all_numeric_macro(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [
                    {
                        "technique": "missing_indicators",
                        "columns": ["ALL_NUMERIC"],
                        "rationale": "Improve stability.",
                    }
                ]
            },
            "critique_packet": {
                "error_modes": [{"id": "fold_instability"}],
                "analysis_summary": "Variance across folds.",
            },
            "experiment_tracker": [],
        }
    )
    valid, errors = validate_iteration_hypothesis_packet(packet)
    assert valid is True, errors
    assert packet.get("action") == "APPLY"
    assert "ALL_NUMERIC" in (packet.get("hypothesis", {}).get("target_columns") or [])


def test_generate_iteration_hypothesis_downgrades_duplicate_to_noop(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    first_packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [{"technique": "missing_indicators", "columns": ["ALL_NUMERIC"]}]
            },
            "critique_packet": {"error_modes": [{"id": "fold_instability"}]},
            "experiment_tracker": [],
        }
    )
    signature = (
        first_packet.get("tracker_context", {}).get("signature")
        if isinstance(first_packet.get("tracker_context"), dict)
        else None
    )
    second_packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 3,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [{"technique": "missing_indicators", "columns": ["ALL_NUMERIC"]}]
            },
            "critique_packet": {"error_modes": [{"id": "fold_instability"}]},
            "experiment_tracker": [{"signature": signature}],
        }
    )
    assert second_packet.get("action") == "NO_OP"
    assert second_packet.get("hypothesis", {}).get("technique") == "NO_OP"

