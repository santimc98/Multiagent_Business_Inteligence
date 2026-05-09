from src.utils.business_impact_context import (
    build_business_impact_context_pack,
    summarize_business_impact_context_pack,
)


def test_business_impact_context_links_metrics_caveats_and_integration() -> None:
    pack = build_business_impact_context_pack(
        business_objective="Prioritize insured portfolios by risk.",
        executive_decision_label="GO_WITH_LIMITATIONS",
        run_summary={
            "status": "completed",
            "run_outcome": "GO_WITH_LIMITATIONS",
            "metrics": {"accuracy_exact": 0.67},
            "warnings": ["temporal stability not testable"],
        },
        metrics_payload={"primary_metric": "qwk", "qwk": 0.78, "mae": 0.42},
        data_adequacy_report={"status": "LIMITED", "reasons": ["missing_temporal_axis"]},
        case_alignment_report={"status": "PASS", "metrics": {"within_band": 0.88}},
        integration_card={
            "business_context": {"decisioning_requirements": {"threshold_policy": "human_defined"}},
            "execution_contract": {"entrypoints": [{"path": "scoring_function.py"}]},
            "output_contract": {"missing_required_artifacts": []},
        },
        model_dependency_context_pack={
            "model_dependency_signals": {
                "top_feature_share": 0.34,
                "top_source_family_share": 0.62,
                "source_family_shares": [{"family": "public_or_legal", "share": 0.62}],
            }
        },
        feature_governance_pack={
            "feature_governance_signals": {
                "semantic_duplicate_groups": [{"columns": ["Sector", "SectorCoface15"]}],
            }
        },
    )

    assert pack["role"] == "business_impact_advisory"
    assert "must not invent" in pack["deterministic_policy"]
    assert pack["decision_context"]["executive_decision_label"] == "GO_WITH_LIMITATIONS"
    assert len(pack["metric_context"]["records"]) >= 3
    areas = {item["area"] for item in pack["business_caveats"]}
    assert "data_adequacy" in areas
    assert "model_dependency" in areas
    assert "feature_governance" in areas
    assert pack["operational_examples"]


def test_business_impact_summary_is_advisory() -> None:
    pack = build_business_impact_context_pack(
        executive_decision_label="GO",
        run_summary={"run_outcome": "GO", "status": "completed"},
        metrics_payload={"auc": 0.91},
    )

    summary = summarize_business_impact_context_pack(pack)

    assert "BUSINESS_IMPACT_CONTEXT_PACK_SUMMARY" in summary
    assert "business_impact_advisory" in summary
    assert "not deterministic deployment policy" in summary
    assert "metric_records_count" in summary
