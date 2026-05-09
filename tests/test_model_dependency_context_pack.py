from src.utils.model_dependency_context import (
    build_model_dependency_context_pack,
    summarize_model_dependency_context_pack,
)


def test_model_dependency_context_measures_feature_and_family_concentration() -> None:
    model_card = {
        "features_used": ["SectorCoface15", "Segmentacion0", "Liquidity"],
        "feature_importances": [
            {"feature": "SectorCoface15", "importance": 0.55},
            {"feature": "Segmentacion0", "importance": 0.25},
            {"feature": "Liquidity", "importance": 0.20},
        ],
    }
    feature_governance_pack = {
        "feature_governance_signals": {
            "semantic_duplicate_groups": [{"columns": ["SectorCoface15", "Sector"]}],
            "high_correlation_pairs": [{"col_a": "A", "col_b": "B", "corr_abs": 0.98}],
        }
    }

    pack = build_model_dependency_context_pack(
        model_card=model_card,
        feature_governance_pack=feature_governance_pack,
    )

    assert pack["role"] == "advisory_context_only"
    assert "must not reject" in pack["deterministic_policy"]
    signals = pack["model_dependency_signals"]
    assert signals["feature_importance_available"] is True
    assert signals["top_feature_share"] == 0.55
    assert signals["top_3_feature_share"] == 1.0
    assert signals["pre_model_semantic_duplicate_groups_count"] == 1
    assert signals["pre_model_high_correlation_pairs_count"] == 1
    assert signals["source_family_shares"][0]["family"] in {
        "commercial_classification",
        "financial_statement",
    }


def test_model_dependency_summary_is_advisory_when_importance_missing() -> None:
    pack = build_model_dependency_context_pack(
        metrics_payload={"features_used": ["risk_score", "risk_band"]},
    )

    summary = summarize_model_dependency_context_pack(pack)

    assert "MODEL_DEPENDENCY_CONTEXT_PACK_SUMMARY" in summary
    assert "advisory_context_only" in summary
    assert "not deterministic pass/fail" in summary
    assert "importance_records_found: 0" in summary
