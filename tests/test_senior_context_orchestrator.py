from src.utils.senior_context_orchestrator import (
    build_senior_context_manifest,
    summarize_senior_context_manifest,
)


def _pack(role: str, policy: str = "This is advisory and must not reject outputs.") -> dict:
    return {"role": role, "deterministic_policy": policy}


def test_senior_context_manifest_reports_full_advisory_coverage() -> None:
    manifest = build_senior_context_manifest(
        packs={
            "data_quality_shape_pack": _pack("advisory_context_only"),
            "feature_governance_pack": _pack("advisory_context_only"),
            "model_dependency_context_pack": _pack("advisory_context_only"),
            "integration_card": _pack("integration_handoff_advisory"),
            "business_impact_context_pack": _pack("business_impact_advisory", "This pack must not invent deployment policy."),
            "report_narrative_contract": _pack("translator_narrative_contract", "This contract does not decide the business outcome."),
        },
        summaries={
            "data_quality_shape_summary": "ok",
            "feature_governance_summary": "ok",
            "model_dependency_context_summary": "ok",
            "integration_card_summary": "ok",
            "business_impact_context_summary": "ok",
            "report_narrative_contract_summary": "ok",
        },
    )

    coverage = manifest["coverage"]
    assert coverage["all_present"] is True
    assert coverage["all_roles_ok"] is True
    assert coverage["all_policies_advisory"] is True
    assert coverage["present_count"] == coverage["expected_count"]


def test_senior_context_manifest_flags_missing_and_non_advisory_policy() -> None:
    manifest = build_senior_context_manifest(
        packs={
            "data_quality_shape_pack": _pack("hard_gate", "must reject when missingness is high"),
        },
        summaries={},
    )

    coverage = manifest["coverage"]
    assert coverage["all_present"] is False
    assert coverage["all_roles_ok"] is False
    assert coverage["all_policies_advisory"] is False
    assert "feature_governance_pack" in coverage["missing"]
    assert coverage["role_mismatches"][0]["id"] == "data_quality_shape_pack"
    assert coverage["policy_warnings"][0]["id"] == "data_quality_shape_pack"


def test_senior_context_manifest_summary_is_compact() -> None:
    manifest = build_senior_context_manifest(packs={}, summaries={})

    summary = summarize_senior_context_manifest(manifest)

    assert "SENIOR_CONTEXT_MANIFEST_SUMMARY" in summary
    assert "senior_context_orchestration_manifest" in summary
    assert "missing" in summary
