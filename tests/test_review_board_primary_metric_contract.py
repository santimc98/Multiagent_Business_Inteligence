from src.graph import graph as graph_mod


def test_extract_primary_metric_for_board_prefers_contract_metric_over_heuristic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "accuracy",
            }
        }
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.93,
            "accuracy": 0.81,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "accuracy"
    assert primary.get("value") == 0.81
    assert primary.get("source") == "state.metrics_report"


def test_extract_primary_metric_for_board_ignores_snapshot_when_contract_metric_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.94,
        },
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "accuracy",
            }
        },
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.94,
            "accuracy": 0.79,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "accuracy"
    assert primary.get("value") == 0.79
    assert primary.get("source") != "primary_metric_snapshot"


def test_extract_primary_metric_for_board_reports_missing_contract_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "normalized_gini",
            }
        }
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.9,
            "accuracy": 0.78,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "normalized_gini"
    assert primary.get("value") is None
    assert primary.get("source") == "contract.primary_metric_missing"
