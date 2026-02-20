from src.agents.reviewer import _deterministic_reviewer_prechecks
from src.graph.graph import _apply_ml_reviewer_output_scope


def test_reviewer_precheck_ignores_data_engineer_outputs() -> None:
    code = "import pandas as pd\ndf = pd.read_csv('data/cleaned_data.csv')\n"
    reviewer_view = {
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/cleaning_manifest.json",
        ]
    }
    evaluation_spec = {
        "required_outputs": [
            {"path": "data/cleaned_data.csv", "owner": "data_engineer"},
            {"path": "data/cleaning_manifest.json", "owner": "data_engineer"},
        ]
    }

    result = _deterministic_reviewer_prechecks(code, evaluation_spec, reviewer_view)

    assert "reviewer_required_outputs_traceability" not in (result.get("failed_gates") or [])
    assert "reviewer_required_outputs_traceability" not in (result.get("hard_failures") or [])


def test_reviewer_precheck_still_requires_ml_outputs() -> None:
    code = "import pandas as pd\ndf = pd.read_csv('data/cleaned_data.csv')\n"
    reviewer_view = {"required_outputs": ["data/metrics.json"]}
    evaluation_spec = {"required_outputs": ["data/metrics.json"]}

    result = _deterministic_reviewer_prechecks(code, evaluation_spec, reviewer_view)

    assert "reviewer_required_outputs_traceability" in (result.get("failed_gates") or [])
    assert "reviewer_required_outputs_traceability" in (result.get("hard_failures") or [])


def test_apply_ml_reviewer_output_scope_filters_de_outputs() -> None:
    contract = {
        "required_outputs": [
            {"path": "data/cleaned_data.csv", "owner": "data_engineer"},
            {"path": "data/cleaning_manifest.json", "owner": "data_engineer"},
            {"path": "data/metrics.json", "owner": "ml_engineer"},
            {"path": "data/submission.csv", "owner": "ml_engineer"},
        ]
    }
    state = {
        "execution_contract": contract,
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
        },
    }
    reviewer_view = {
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/metrics.json",
            "data/submission.csv",
        ],
        "verification": {"required_outputs": ["data/cleaned_data.csv", "data/metrics.json"]},
    }
    evaluation_spec = {"required_outputs": ["data/cleaning_manifest.json", "data/metrics.json"]}

    scoped_view, scoped_eval = _apply_ml_reviewer_output_scope(
        contract,
        state,
        reviewer_view,
        evaluation_spec,
    )

    assert scoped_view.get("required_outputs") == ["data/metrics.json", "data/submission.csv"]
    assert (scoped_view.get("verification") or {}).get("required_outputs") == [
        "data/metrics.json",
        "data/submission.csv",
    ]
    assert (scoped_eval or {}).get("required_outputs") == ["data/metrics.json", "data/submission.csv"]
