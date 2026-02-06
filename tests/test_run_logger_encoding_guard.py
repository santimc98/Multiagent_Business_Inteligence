import json
from pathlib import Path

from src.utils.run_logger import init_run_log, log_run_event


def test_log_run_event_sanitizes_payload_and_emits_encoding_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "enc_guard_001"
    init_run_log(run_id)

    log_run_event(
        run_id,
        "translator_output",
        {
            "text": "Necesita revisi\u00c3\u00b3n manual",
            "nested": ["d\u00c3\u00adgitos", {"label": "se\u00c3\u00b1ales"}],
        },
    )

    event_path = Path("logs") / f"run_{run_id}.jsonl"
    entries = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    target = next(item for item in entries if item.get("event") == "translator_output")

    assert target["payload"]["text"] == "Necesita revisi\u00f3n manual"
    assert target["payload"]["nested"][0] == "d\u00edgitos"
    assert target["payload"]["nested"][1]["label"] == "se\u00f1ales"
    assert int(target.get("encoding_guard", {}).get("strings_changed", 0)) >= 3
    assert int(target.get("encoding_guard", {}).get("mojibake_hits", 0)) >= 3
