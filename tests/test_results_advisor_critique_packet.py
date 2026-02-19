from src.agents.results_advisor import ResultsAdvisorAgent
from src.utils.actor_critic_schemas import validate_advisor_critique_packet


def test_generate_critique_packet_includes_holdout_validation_signals() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    packet = advisor.generate_critique_packet(
        {
            "run_id": "run_test",
            "iteration": 1,
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "min_delta": 0.0005,
            "baseline_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.801,
                    "cv_std": 0.012,
                    "holdout_roc_auc": 0.796,
                }
            },
            "candidate_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.801,
                    "cv_std": 0.012,
                    "holdout_roc_auc": 0.796,
                }
            },
            "active_gates_context": ["required_artifacts_present", "target_variance_guard"],
            "dataset_profile": {"n_rows": 1200},
        }
    )
    valid, errors = validate_advisor_critique_packet(packet)
    assert valid is True, errors
    validation = packet.get("validation_signals", {})
    assert validation.get("validation_mode") in {"holdout", "cv_and_holdout"}
    assert packet.get("strictly_no_code_advice") is True

