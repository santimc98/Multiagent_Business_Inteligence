from __future__ import annotations

from typing import Any, Dict, List


_EXPECTED_PACKS = [
    {
        "id": "data_quality_shape_pack",
        "allowed_roles": {"advisory_context_only"},
        "summary_key": "data_quality_shape_summary",
        "purpose": "data quality shape, zero-vs-null, dispersion, concentration",
    },
    {
        "id": "feature_governance_pack",
        "allowed_roles": {"advisory_context_only"},
        "summary_key": "feature_governance_summary",
        "purpose": "semantic duplicates, correlations, feature families",
    },
    {
        "id": "model_dependency_context_pack",
        "allowed_roles": {"advisory_context_only"},
        "summary_key": "model_dependency_context_summary",
        "purpose": "feature/family dominance and source dependency",
    },
    {
        "id": "integration_card",
        "allowed_roles": {"integration_handoff_advisory"},
        "summary_key": "integration_card_summary",
        "purpose": "production handoff, inputs, outputs, entrypoints",
    },
    {
        "id": "business_impact_context_pack",
        "allowed_roles": {"business_impact_advisory"},
        "summary_key": "business_impact_context_summary",
        "purpose": "business impact, caveats, operational examples",
    },
    {
        "id": "report_narrative_contract",
        "allowed_roles": {"translator_narrative_contract"},
        "summary_key": "report_narrative_contract_summary",
        "purpose": "translator coverage and evidence discipline",
    },
]


def _is_advisory_policy(text: Any) -> bool:
    policy = str(text or "").lower()
    if not policy:
        return False
    forbidden = ("hard gate", "auto reject", "automatic rejection", "must reject", "block the run")
    if any(token in policy for token in forbidden):
        return False
    return any(
        token in policy
        for token in (
            "must not reject",
            "does not decide",
            "not deterministic",
            "not decide",
            "advisory",
            "must not invent",
        )
    )


def build_senior_context_manifest(
    *,
    packs: Dict[str, Any] | None = None,
    summaries: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    packs = packs if isinstance(packs, dict) else {}
    summaries = summaries if isinstance(summaries, dict) else {}
    entries: List[Dict[str, Any]] = []
    missing: List[str] = []
    role_mismatches: List[Dict[str, Any]] = []
    policy_warnings: List[Dict[str, Any]] = []

    for spec in _EXPECTED_PACKS:
        pack_id = str(spec["id"])
        pack = packs.get(pack_id)
        present = isinstance(pack, dict) and bool(pack)
        if not present:
            missing.append(pack_id)
            entries.append(
                {
                    "id": pack_id,
                    "present": False,
                    "purpose": spec["purpose"],
                    "role": None,
                    "summary_available": bool(str(summaries.get(spec["summary_key"]) or "").strip()),
                    "advisory_policy_ok": False,
                }
            )
            continue
        role = str(pack.get("role") or "")
        role_ok = role in set(spec["allowed_roles"])
        if not role_ok:
            role_mismatches.append(
                {
                    "id": pack_id,
                    "role": role,
                    "expected": sorted(spec["allowed_roles"]),
                }
            )
        policy = pack.get("deterministic_policy")
        advisory_policy_ok = _is_advisory_policy(policy)
        if not advisory_policy_ok:
            policy_warnings.append(
                {
                    "id": pack_id,
                    "issue": "missing_or_non_advisory_policy",
                    "policy_excerpt": str(policy or "")[:240],
                }
            )
        entries.append(
            {
                "id": pack_id,
                "present": True,
                "purpose": spec["purpose"],
                "role": role,
                "role_ok": role_ok,
                "summary_key": spec["summary_key"],
                "summary_available": bool(str(summaries.get(spec["summary_key"]) or "").strip()),
                "advisory_policy_ok": advisory_policy_ok,
            }
        )

    return {
        "schema_version": "1.0",
        "role": "senior_context_orchestration_manifest",
        "deterministic_policy": (
            "This manifest verifies context availability and advisory policy. It must not decide, reject, "
            "or approve any agent output."
        ),
        "entries": entries,
        "coverage": {
            "expected_count": len(_EXPECTED_PACKS),
            "present_count": sum(1 for item in entries if item.get("present")),
            "missing": missing,
            "role_mismatches": role_mismatches,
            "policy_warnings": policy_warnings,
            "all_present": not missing,
            "all_roles_ok": not role_mismatches,
            "all_policies_advisory": not policy_warnings,
        },
        "senior_usage_guidance": [
            "Use this manifest to confirm the report context is complete before relying on a translator output.",
            "Missing packs are context gaps, not automatic failures.",
            "Role or policy mismatches indicate architectural regression because advisory packs may have become deterministic gates.",
        ],
    }


def summarize_senior_context_manifest(manifest: Dict[str, Any] | None) -> str:
    if not isinstance(manifest, dict) or not manifest:
        return ""
    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    lines = [
        "SENIOR_CONTEXT_MANIFEST_SUMMARY:",
        "- role: senior_context_orchestration_manifest; context coverage/advisory-policy regression guard, not outcome decision logic",
        f"- present_count: {coverage.get('present_count')}/{coverage.get('expected_count')}; all_present={coverage.get('all_present')}; all_roles_ok={coverage.get('all_roles_ok')}; all_policies_advisory={coverage.get('all_policies_advisory')}",
    ]
    missing = coverage.get("missing")
    if missing:
        lines.append(f"- missing: {missing}")
    role_mismatches = coverage.get("role_mismatches")
    if role_mismatches:
        lines.append(f"- role_mismatches: {role_mismatches}")
    policy_warnings = coverage.get("policy_warnings")
    if policy_warnings:
        lines.append(f"- policy_warnings: {policy_warnings}")
    lines.append("- entries:")
    for item in entries:
        if isinstance(item, dict):
            lines.append(
                "  "
                + str(
                    {
                        "id": item.get("id"),
                        "present": item.get("present"),
                        "role": item.get("role"),
                        "summary_available": item.get("summary_available"),
                        "advisory_policy_ok": item.get("advisory_policy_ok"),
                    }
                )
            )
    return "\n".join(lines)
