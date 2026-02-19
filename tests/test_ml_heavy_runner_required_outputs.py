from src.graph.graph import _resolve_ml_heavy_runner_required_outputs


def test_resolve_ml_heavy_runner_required_outputs_excludes_de_inputs() -> None:
    contract = {
        "required_outputs": [
            {"path": "data/cleaned_data.csv", "owner": "data_engineer"},
            {"path": "data/cleaning_manifest.json", "owner": "data_engineer"},
            {"path": "data/metrics.json", "owner": "ml_engineer"},
            "data/submission.csv",
            "data\\submission.csv",
            "data/cleaned_full.csv",
        ],
        "artifact_requirements": {
            "required_files": ["data/metrics.json", "data/submission.csv"],
        },
    }
    state = {
        "execution_contract": contract,
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
            "manifest_path": "data/cleaning_manifest.json",
        },
    }

    outputs = _resolve_ml_heavy_runner_required_outputs(contract, state)

    assert outputs == ["data/metrics.json", "data/submission.csv"]
    assert "data/cleaned_data.csv" not in outputs
    assert "data/cleaning_manifest.json" not in outputs
    assert "data/cleaned_full.csv" not in outputs


def test_resolve_ml_heavy_runner_required_outputs_fallback_to_artifact_requirements() -> None:
    contract = {
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/cleaning_manifest.json",
            "data/cleaned_full.csv",
        ],
        "artifact_requirements": {
            "required_files": ["data/metrics.json", {"path": "data/submission.csv"}],
        },
    }
    state = {
        "execution_contract": contract,
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
        },
    }

    outputs = _resolve_ml_heavy_runner_required_outputs(contract, state)

    assert outputs == ["data/metrics.json", "data/submission.csv"]
    assert "data/cleaned_data.csv" not in outputs
    assert "data/cleaning_manifest.json" not in outputs
    assert "data/cleaned_full.csv" not in outputs
