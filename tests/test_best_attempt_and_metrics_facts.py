import json
import os
import time

from src.graph import graph as graph_mod


def test_build_review_board_facts_prefers_best_output_contract_and_insights_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    insights = {
        "metrics_summary": [
            {"metric": "model_performance.Normalized Gini", "value": 0.3125},
            {"metric": "model_performance.ROC-AUC", "value": 0.6562},
        ]
    }
    with open("data/insights.json", "w", encoding="utf-8") as handle:
        json.dump(insights, handle)

    state = {
        "execution_contract": {"evaluation_spec": {"objective_type": "classification"}},
        "output_contract_report": {
            "overall_status": "error",
            "present": ["data/cleaned_data.csv"],
            "missing": ["reports/evaluation_metrics.json"],
        },
        "best_attempt_output_contract_report": {
            "overall_status": "ok",
            "present": ["reports/evaluation_metrics.json"],
            "missing": [],
        },
    }

    facts = graph_mod._build_review_board_facts(state)

    assert facts["output_contract"]["missing_required_artifacts"] == []
    assert facts["metrics"]["primary"]["name"] in {"Normalized Gini", "ROC-AUC"}
    assert facts["metrics"]["primary"]["value"] is not None
    assert facts["metrics"]["primary"]["source"] in {"data/insights.json", "metrics.normalized"}


def test_finalize_heavy_execution_updates_best_attempt_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("data/cleaned_data.csv", "w", encoding="utf-8") as handle:
        handle.write("x,target\n1,0\n")
    with open("reports/evaluation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"model_performance": {"accuracy": 0.9}}, handle)

    state = {
        "execution_attempt": 0,
        "ml_data_path": "data/cleaned_data.csv",
    }
    contract = {
        "required_outputs": ["reports/evaluation_metrics.json"],
    }

    result = graph_mod._finalize_heavy_execution(
        state=state,
        output="Execution completed successfully.",
        exec_start_ts=time.time() - 1.0,
        contract=contract,
        eval_spec={},
        csv_sep=",",
        csv_decimal=".",
        csv_encoding="utf-8",
        counters={},
        run_id=None,
        attempt_id=2,
        visuals_missing=False,
    )

    assert result["last_attempt_valid"] is True
    assert result["best_attempt_id"] == 2
    assert result["best_attempt_output_contract_report"]["missing"] == []
    assert os.path.isdir(result["best_attempt_dir"])
