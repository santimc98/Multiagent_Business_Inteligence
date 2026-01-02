import json
import os
import time
from pathlib import Path

from src.utils.run_bundle import init_run_bundle, write_run_manifest, copy_run_artifacts
from src.utils.run_logger import init_run_log, log_run_event


def test_run_bundle_creates_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

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
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    contracts_dir = Path(run_dir) / "contracts"
    artifacts_dir = Path(run_dir) / "artifacts" / "data"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contract = {"required_outputs": ["data/metrics.json"]}
    (contracts_dir / "execution_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text("{}", encoding="utf-8")
    manifest_path = write_run_manifest(run_id, state)
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["input"]["path"] == str(csv_path)
    assert "data/metrics.json" in manifest["required_outputs"]
    assert "data/metrics.json" in manifest["produced_outputs"]


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


def test_manifest_no_ml_outputs_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run9999"
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "run_start_ts": "2025-01-01T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    contracts_dir = Path(run_dir) / "contracts"
    artifacts_dir = Path(run_dir) / "artifacts" / "data"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contract = {"required_outputs": ["data/cleaned_data.csv", "data/metrics.json"]}
    (contracts_dir / "execution_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifacts_dir / "cleaned_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    manifest_path = write_run_manifest(run_id, state, status_final="FAIL")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert "data/metrics.json" not in manifest["produced_outputs"]


def test_copy_run_artifacts_filters_by_mtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run2222"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True, exist_ok=True)
    old_file = source_dir / "old.txt"
    new_file = source_dir / "new.txt"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")

    now = time.time()
    os.utime(old_file, (now - 10, now - 10))
    os.utime(new_file, (now + 2, now + 2))

    copy_run_artifacts(run_id, [str(source_dir)], since_epoch=now - 1)

    dest_old = Path(run_dir) / "artifacts" / "data" / "old.txt"
    dest_new = Path(run_dir) / "artifacts" / "data" / "new.txt"
    assert not dest_old.exists()
    assert dest_new.exists()
