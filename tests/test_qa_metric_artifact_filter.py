from src.agents.qa_reviewer import _collect_metric_artifact_paths


def test_collect_metric_artifact_paths_excludes_inference_benchmark_from_metric_facts():
    paths = _collect_metric_artifact_paths(
        {
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "intent": "metrics"},
                {"path": "artifacts/ml/inference_benchmark.json", "intent": "benchmark"},
                {"path": "artifacts/ml/model_card.json", "intent": "model card"},
            ]
        },
        subject_required_outputs=[],
        qa_required_outputs=[],
    )

    assert "artifacts/ml/cv_metrics.json" in paths
    assert "artifacts/ml/inference_benchmark.json" not in paths
    assert "artifacts/ml/model_card.json" not in paths
