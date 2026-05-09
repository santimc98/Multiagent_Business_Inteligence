from src.utils.feature_governance import (
    build_feature_governance_pack,
    summarize_feature_governance_pack,
)


def test_feature_governance_detects_semantic_duplicate_groups_and_correlations() -> None:
    dataset_profile = {
        "columns": [
            "SectorCoface15",
            "Sector",
            "Segmentacion0",
            "Segmento",
            "Ventas12M",
            "VentasUltimoAnio",
        ]
    }
    data_profile = {
        "basic_stats": {"columns": dataset_profile["columns"]},
        "multicollinearity_pairs_high": [
            {"col_a": "Ventas12M", "col_b": "VentasUltimoAnio", "corr_abs": 0.982},
        ],
        "feature_target_associations": [
            {"column": "SectorCoface15", "score": 0.42, "method": "chi2_cramers_v"},
            {"column": "Segmentacion0", "score": 0.21, "method": "chi2_cramers_v"},
        ],
    }

    pack = build_feature_governance_pack(dataset_profile, data_profile=data_profile)

    assert pack["role"] == "advisory_context_only"
    assert "must not reject" in pack["deterministic_policy"]
    signals = pack["feature_governance_signals"]
    duplicate_groups = signals["semantic_duplicate_groups"]
    assert any({"SectorCoface15", "Sector"}.issubset(set(group["columns"])) for group in duplicate_groups)
    assert any({"Segmentacion0", "Segmento"}.issubset(set(group["columns"])) for group in duplicate_groups)
    assert signals["high_correlation_pairs"][0]["corr_abs"] == 0.982
    assert signals["target_association_concentration"]["available"] is True


def test_feature_governance_summary_is_advisory_and_compact() -> None:
    pack = build_feature_governance_pack(
        {"columns": ["risk_score", "risk_band", "customer_id"]},
        data_profile={
            "multicollinearity_pairs_high": [
                {"col_a": "risk_score", "col_b": "risk_band", "corr_abs": 0.97},
            ]
        },
        column_sets={"sets": [{"name": "risk", "count": 2, "selector": {"type": "prefix", "value": "risk_"}}]},
    )

    summary = summarize_feature_governance_pack(pack)

    assert "FEATURE_GOVERNANCE_PACK_SUMMARY" in summary
    assert "advisory_context_only" in summary
    assert "not deterministic pass/fail" in summary
    assert "high_correlation_pairs=1" in summary
