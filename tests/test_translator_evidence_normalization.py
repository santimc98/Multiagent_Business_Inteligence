import re

from src.agents.business_translator import _ensure_evidence_section


def test_ensure_evidence_section_rebuilds_single_canonical_block():
    report = """## Resumen\n\nTexto base.\n\n## Evidencia usada\n\nevidence:\n{claim: \"legacy\", source: \"inline_token\"}\n\nArtifacts:\n- data/old_a.csv\n\n## Evidencia usada\n\nevidence:\n{claim: \"legacy2\", source: \"inline_token_2\"}\n\nArtifacts:\n- data/old_b.csv\n"""
    evidence_paths = ["data/metrics.json", "data/scored_rows.csv", "data/metrics.json"]

    normalized = _ensure_evidence_section(report, evidence_paths)

    assert len(re.findall(r"(?im)^##\s+Evidencia usada\s*$", normalized)) == 1
    assert normalized.count("Artifacts:") == 1
    assert "- data/metrics.json" in normalized
    assert "- data/scored_rows.csv" in normalized
    assert normalized.count("- data/metrics.json") == 1
    assert "{claim:" in normalized
    assert "source:" in normalized


def test_ensure_evidence_section_repairs_text_before_rebuild():
    report = "## Resumen\n\nNecesita revisi\u00c3\u00b3n manual\n"
    normalized = _ensure_evidence_section(report, ["data/summary.md"])
    assert "Necesita revisi\u00f3n manual" in normalized
