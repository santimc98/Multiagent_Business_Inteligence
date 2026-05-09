import json

from src.utils.output_contract import build_output_contract_report


def test_output_contract_infers_machine_checkable_artifact_interfaces(tmp_path):
    ml_dir = tmp_path / "artifacts" / "ml"
    ml_dir.mkdir(parents=True)
    (ml_dir / "predictions.csv").write_text("EntityId,riim_pred\n1,4\n", encoding="utf-8")
    (ml_dir / "feature_drift_baseline.json").write_text(
        json.dumps({"feature_a": {"mean": 1.0, "median": 1.0}}),
        encoding="utf-8",
    )
    (ml_dir / "scoring_function.py").write_text(
        "def score(df_features):\n    return df_features\n",
        encoding="utf-8",
    )

    contract = {
        "ml_engineer": {
            "required_outputs": [
                {
                    "path": "artifacts/ml/predictions.csv",
                    "required": True,
                    "description": "CSV with columns EntityId, RIIM10, riim10_pred, and probability columns.",
                },
                {
                    "path": "artifacts/ml/feature_drift_baseline.json",
                    "required": True,
                    "description": "JSON with mean, p25, median, and p75 per feature.",
                },
                {
                    "path": "artifacts/ml/scoring_function.py",
                    "required": True,
                    "description": "Pure Python predict(df_features) returns riim10_pred and riim10_pred_proba_levelN.",
                },
            ]
        }
    }

    report = build_output_contract_report(contract, work_dir=str(tmp_path))

    issues = "\n".join(report.get("schema_issues") or [])
    assert "predictions.csv missing required columns" in issues
    assert "RIIM10" in issues
    assert "riim10_pred" in issues
    assert "feature_drift_baseline.json missing required JSON keys" in issues
    assert "p25" in issues
    assert "p75" in issues
    assert "scoring_function.py missing expected function(s): predict" in issues
    assert report["artifact_requirements_report"]["schema_interface_report"]["checked"]


def test_output_contract_schema_inference_ignores_prose_tokens(tmp_path):
    ml_dir = tmp_path / "artifacts" / "ml"
    ml_dir.mkdir(parents=True)
    (ml_dir / "predictions.csv").write_text(
        "EntityId,CorporationId,MonthId,riim_actual,riim_pred\n1,A,2025-06-01,4,5\n",
        encoding="utf-8",
    )
    (ml_dir / "portfolio_aggregation.csv").write_text(
        "CorporationId,riim_weighted_real,riim_weighted_pred,abs_deviation\nA,4.1,4.2,0.1\n",
        encoding="utf-8",
    )
    (ml_dir / "inference_benchmark.json").write_text(
        json.dumps({"batch_1000": {"mean_latency_ms": 3.4, "p95_latency_ms": 4.6}}),
        encoding="utf-8",
    )

    contract = {
        "required_outputs": [
            {
                "path": "artifacts/ml/predictions.csv",
                "required": True,
                "description": "Row-level predictions on holdout (Jun 2025) and scoring partitions with EntityId, CorporationId, MonthId, RIIM10, riim10_pred, and probability columns.",
            },
            {
                "path": "artifacts/ml/portfolio_aggregation.csv",
                "required": True,
                "description": "One row per CorporationId with real and predicted sales-weighted mean RIIM, absolute deviation, and deviation flag.",
            },
            {
                "path": "artifacts/ml/inference_benchmark.json",
                "required": True,
                "description": "Inference latency (mean and p95) for 1K debtor batches.",
            },
        ]
    }

    report = build_output_contract_report(contract, work_dir=str(tmp_path))
    issues = "\n".join(report.get("schema_issues") or [])

    assert "Jun" not in issues
    assert "One" not in issues
    assert "portfolio_aggregation.csv missing required columns" not in issues
    assert "inference_benchmark.json missing required JSON keys" not in issues
    assert "RIIM10" in issues
    assert "riim10_pred" in issues
