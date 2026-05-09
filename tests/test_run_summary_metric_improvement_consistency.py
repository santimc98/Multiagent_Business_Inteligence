import json
import os

from src.utils.governance import build_run_summary


def test_build_run_summary_tracks_metric_improvement_with_artifact_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open("data/review_board_verdict.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric_round_finalization": {
                    "metric_name": "cv_roc_auc",
                    "kept": "improved",
                    "baseline_metric": 0.8011,
                    "candidate_metric": 0.8030,
                    "final_metric": 0.8030,
                    "force_finalize_reason": "",
                }
            },
            handle,
            indent=2,
        )

    with open("data/metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"cv_roc_auc": 0.8030}, handle, indent=2)

    summary = build_run_summary({"review_verdict": "APPROVED", "ml_improvement_kept": "improved"})
    metric_improvement = summary.get("metric_improvement", {})

    assert metric_improvement.get("kept") == "improved"
    assert metric_improvement.get("metric_name") == "cv_roc_auc"
    assert metric_improvement.get("final_metric_artifact") == 0.803


def test_build_run_summary_does_not_no_go_restored_metric_incumbent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open("data/output_contract_report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "overall_status": "ok",
                "missing": [],
                "qa_gate_failures": [],
                "blocking_qa_gate_failures": [],
            },
            handle,
            indent=2,
        )
    with open("data/review_board_verdict.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "NEEDS_IMPROVEMENT",
                "final_review_verdict": "NEEDS_IMPROVEMENT",
                "deterministic_blockers": ["result_evaluator_failed_gate:runtime_failure"],
                "metric_round_finalization": {
                    "metric_name": "qwk",
                    "kept": "baseline",
                    "baseline_metric": 0.7626,
                    "candidate_metric": 0.7626,
                    "final_metric": 0.7626,
                },
            },
            handle,
            indent=2,
        )
    with open("data/metric_loop_state.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "target": {"name": "qwk"},
                "final": {
                    "label": "baseline",
                    "metric_value": 0.7626,
                    "review_verdict": "APPROVE_WITH_WARNINGS",
                },
                "selection": {"selected_label": "baseline"},
            },
            handle,
            indent=2,
        )
    with open("data/metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"holdout": {"qwk": 0.7626}}, handle, indent=2)

    summary = build_run_summary(
        {
            "review_verdict": "NEEDS_IMPROVEMENT",
            "last_successful_review_verdict": "APPROVE_WITH_WARNINGS",
            "last_gate_context": {
                "failed_gates": [
                    "feature_drift_baseline_scope",
                    "contract_required_artifacts_missing",
                ],
                "hard_failures": ["feature_drift_baseline_scope"],
            },
            "runtime_fix_terminal": True,
            "hard_failures": [
                "feature_drift_baseline_scope",
                "result_evaluator_failed_gate:runtime_failure",
            ],
            "ml_improvement_kept": "baseline",
        }
    )

    assert summary["status"] == "APPROVE_WITH_WARNINGS"
    assert summary["run_outcome"] == "GO_WITH_LIMITATIONS"
    assert "feature_drift_baseline_scope" not in summary["failed_gates"]
    assert "result_evaluator_failed_gate:runtime_failure" not in summary["hard_failures"]
