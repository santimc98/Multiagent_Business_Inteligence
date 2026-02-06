from src.utils.text_encoding import (
    has_mojibake,
    sanitize_text,
    sanitize_text_payload_with_stats,
)


def test_sanitize_text_repairs_single_layer_mojibake_spanish():
    raw = "Lectura de d\u00c3\u00adgitos con revisi\u00c3\u00b3n manual"
    fixed = sanitize_text(raw)
    assert fixed == "Lectura de d\u00edgitos con revisi\u00f3n manual"
    assert has_mojibake(fixed) is False


def test_sanitize_text_repairs_double_layer_mojibake_spanish():
    raw = "Lectura de d\u00c3\u0192\u00c2\u00adgitos con revisi\u00c3\u0192\u00c2\u00b3n manual"
    fixed = sanitize_text(raw)
    assert fixed == "Lectura de d\u00edgitos con revisi\u00f3n manual"
    assert has_mojibake(fixed) is False


def test_sanitize_text_payload_repairs_nested_strings_and_stats():
    payload = {
        "objective": "Automatizar revisi\u00c3\u00b3n",
        "nested": [{"text": "d\u00c3\u00adgitos"}, "se\u00c3\u00b1ales"],
    }
    fixed, stats = sanitize_text_payload_with_stats(payload)
    assert fixed["objective"] == "Automatizar revisi\u00f3n"
    assert fixed["nested"][0]["text"] == "d\u00edgitos"
    assert fixed["nested"][1] == "se\u00f1ales"
    assert int(stats.get("strings_changed", 0)) >= 3
    assert int(stats.get("mojibake_hits", 0)) >= 3
