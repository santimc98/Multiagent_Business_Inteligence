import json
from pathlib import Path

from src.utils.run_bundle import init_run_bundle, write_run_manifest
from src.utils.run_logger import init_run_log, log_run_event


def test_run_bundle_creates_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    contract = {"required_outputs": ["data/metrics.json"]}
    (data_dir / "execution_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (data_dir / "artifact_index.json").write_text(json.dumps([{"path": "data/metrics.json"}]), encoding="utf-8")
    (data_dir / "output_contract_report.json").write_text(
        json.dumps({"missing": []}), encoding="utf-8"
    )
    (data_dir / "run_summary.json").write_text(
        json.dumps({"status": "APPROVED", "failed_gates": []}), encoding="utf-8"
    )

    run_id = "run1234"
    state = {
        "run_id": run_id,
        "run_start_ts": "2025-01-01T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "agent_models": {"steward": "test-model"},
    }
    init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    manifest_path = write_run_manifest(run_id, state)
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["input"]["path"] == str(csv_path)
    assert "data/metrics.json" in manifest["required_outputs"]


def test_events_jsonl_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run5678"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)
    init_run_log(run_id, {"note": "test"})
    log_run_event(run_id, "test_event", {"ok": True})
    events_path = Path(run_dir) / "events.jsonl"
    assert events_path.exists()
    content = events_path.read_text(encoding="utf-8")
    assert "test_event" in content
