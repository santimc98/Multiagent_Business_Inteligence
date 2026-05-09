from src.utils.report_narrative_contract import (
    build_report_narrative_contract,
    summarize_report_narrative_contract,
)


def test_report_narrative_contract_maps_available_packs_to_topics() -> None:
    contract = build_report_narrative_contract(
        data_quality_shape_pack={"shape_signals": {"counts": {"high_missingness": 2}}},
        feature_governance_pack={"feature_governance_signals": {"semantic_duplicate_groups": [{}]}},
        model_dependency_context_pack={"model_dependency_signals": {"top_feature_share": 0.4}},
        integration_card={"input_contract": {"feature_count": 3}},
        business_impact_context_pack={"decision_context": {"executive_decision_label": "GO"}},
        qa_review_signals={"status_is_warning": True, "explicit_warnings": ["calibration caveat"]},
    )

    topic_ids = [item["id"] for item in contract["required_topics"]]

    assert "data_quality_characterization" in topic_ids
    assert "feature_governance" in topic_ids
    assert "model_dependency_and_dominance" in topic_ids
    assert "integration_readiness" in topic_ids
    assert "business_impact_interpretation" in topic_ids
    assert "qa_material_caveats" in topic_ids
    assert "does not decide the business outcome" in contract["deterministic_policy"]


def test_report_narrative_contract_summary_is_compact() -> None:
    contract = build_report_narrative_contract(
        integration_card={"output_contract": {"required_artifact_count": 2}},
    )

    summary = summarize_report_narrative_contract(contract)

    assert "REPORT_NARRATIVE_CONTRACT" in summary
    assert "translator_narrative_contract" in summary
    assert "integration_readiness" in summary
