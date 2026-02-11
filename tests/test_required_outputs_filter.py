from src.graph.graph import _looks_like_filesystem_path, _resolve_required_outputs


def test_looks_like_filesystem_path():
    assert _looks_like_filesystem_path("Priority Rank") is False
    assert _looks_like_filesystem_path("data/metrics.json") is True


def test_resolve_required_outputs_filters_conceptual():
    contract = {
        "evaluation_spec": {
            "required_outputs": [
                "Priority Rank",
                "data/metrics.json",
            ]
        }
    }
    state = {}
    outputs = _resolve_required_outputs(contract, state)

    assert outputs == ["data/metrics.json"]
    reporting = state.get("reporting_requirements", {})
    conceptual = reporting.get("conceptual_outputs", [])
    assert "Priority Rank" in conceptual


def test_resolve_required_outputs_prefers_contract_outputs_as_source_of_truth():
    contract = {
        "required_outputs": ["data/metrics.json"],
        "visualization_requirements": {
            "required": True,
            "required_plots": [{"name": "confidence_distribution"}],
            "outputs_dir": "static/plots",
        },
    }
    outputs = _resolve_required_outputs(contract, {})

    assert outputs == ["data/metrics.json"]


def test_resolve_required_outputs_falls_back_to_visualization_outputs_when_contract_missing():
    contract = {
        "visualization_requirements": {
            "required": True,
            "required_plots": [{"name": "confidence_distribution"}],
            "outputs_dir": "static/plots",
        },
    }

    outputs = _resolve_required_outputs(contract, {})

    assert "static/plots/confidence_distribution.png" in outputs
