import json
from pathlib import Path

from src.graph.graph import (
    check_evaluation,
    _is_improvement,
    _should_run_metric_improvement_round,
    _snapshot_ml_outputs,
    _restore_ml_outputs,
)


def test_should_run_metric_improvement_round_defaults_to_true_after_baseline_approved() -> None:
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }
    contract = {}
    assert _should_run_metric_improvement_round(state, contract) is True


def test_should_run_metric_improvement_round_requires_real_reviewer_pair_approval() -> None:
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "REJECTED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }
    assert _should_run_metric_improvement_round(state, {}) is False


def test_is_improvement_respects_min_delta_threshold() -> None:
    assert _is_improvement(0.8000, 0.8003, True, 0.0005) is False
    assert _is_improvement(0.8000, 0.8010, True, 0.0005) is True


def test_snapshot_and_restore_ml_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("artifacts/data/metrics.json")
    submission_path = Path("artifacts/data/submission.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_metrics = {"roc_auc": 0.81234}
    baseline_submission = "id,pred\n1,0.2\n"
    metrics_path.write_text(json.dumps(baseline_metrics), encoding="utf-8")
    submission_path.write_text(baseline_submission, encoding="utf-8")

    output_paths = [str(metrics_path).replace("\\", "/"), str(submission_path).replace("\\", "/")]
    snapshot_dir = Path("work/ml_baseline_snapshot")
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    metrics_path.write_text(json.dumps({"roc_auc": 0.70001}), encoding="utf-8")
    submission_path.write_text("id,pred\n1,0.9\n", encoding="utf-8")

    _restore_ml_outputs(snapshot_dir, output_paths)

    restored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    restored_submission = submission_path.read_text(encoding="utf-8")
    assert restored_metrics == baseline_metrics
    assert restored_submission == baseline_submission


def test_check_evaluation_restores_baseline_when_improvement_is_below_delta(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {"roc_auc": 0.8000}
    metrics_path.write_text(json.dumps(baseline), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    # Candidate round result (approved but below min_delta)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8003}), encoding="utf-8")
    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "feedback_history": [],
    }

    route = check_evaluation(state)
    restored = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert route == "approved"
    assert state.get("ml_improvement_kept") == "baseline"
    assert restored == baseline


def test_check_evaluation_logs_metric_improvement_round_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    events = []
    from src.graph import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "log_run_event",
        lambda run_id, event_type, payload, log_dir="logs": events.append((run_id, event_type, payload)),
    )
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)

    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8006}), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    state = {
        "run_id": "run_improvement_trace",
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "feedback_history": [],
    }

    route = check_evaluation(state)

    assert route == "approved"
    event_types = [evt[1] for evt in events]
    assert "metric_improvement_round_complete" in event_types
