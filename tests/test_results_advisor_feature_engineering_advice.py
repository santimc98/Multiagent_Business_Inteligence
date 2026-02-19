from src.agents.results_advisor import ResultsAdvisorAgent


def test_generate_feature_engineering_advice_uses_contract_plan() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.7123}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {
                "techniques": [{"technique": "interaction", "columns": ["x", "y"]}],
                "notes": "Mantener leakage guard",
            },
            "baseline_ml_script_snippet": "def train():\n  pass\n",
            "dataset_profile": {},
            "column_roles": {},
        }
    )
    assert "interaction" in advice.lower()
    assert "build_features" in advice
    assert "Mantener leakage guard".lower() in advice.lower()


def test_generate_feature_engineering_advice_fallback_is_universal_and_bounded() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.61}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {"techniques": []},
            "dataset_profile": {
                "features_with_nulls": ["a"],
                "high_cardinality_columns": ["cat_col"],
            },
        }
    )
    lines = [line for line in advice.splitlines() if line.strip()]
    assert any("missing indicators" in line.lower() for line in lines)
    assert any("categorias raras" in line.lower() for line in lines)
    assert len(lines) <= 12
