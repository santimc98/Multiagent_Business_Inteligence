import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_heavy_train_module():
    # Provide a lightweight stub when google cloud client isn't installed.
    if "google.cloud.storage" not in sys.modules:
        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        cloud_mod = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
        storage_mod = sys.modules.setdefault("google.cloud.storage", types.ModuleType("google.cloud.storage"))

        class _DummyClient:
            pass

        storage_mod.Client = _DummyClient
        setattr(cloud_mod, "storage", storage_mod)
        setattr(google_mod, "cloud", cloud_mod)

    module_path = Path(__file__).resolve().parents[1] / "cloudrun" / "heavy_runner" / "heavy_train.py"
    spec = importlib.util.spec_from_file_location("heavy_train_module_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_resolve_execute_code_mode_for_data_engineer():
    heavy_train = _load_heavy_train_module()
    mode, required, skip_paths = heavy_train._resolve_execute_code_mode(
        {
            "mode": "data_engineer_cleaning",
            "required_outputs": ["data/cleaned_data.csv", "data/cleaning_manifest.json"],
        }
    )
    assert mode == "data_engineer_cleaning"
    assert "data/cleaned_data.csv" in required
    assert "data/cleaning_manifest.json" in required
    assert "data/cleaned_data.csv" not in skip_paths


def test_resolve_execute_code_mode_for_data_engineer_legacy_alias():
    heavy_train = _load_heavy_train_module()
    mode, required, skip_paths = heavy_train._resolve_execute_code_mode(
        {
            "mode": "data_engineer",
            "required_outputs": ["data/cleaned_data.csv", "data/cleaning_manifest.json"],
        }
    )
    assert mode == "data_engineer_cleaning"
    assert "data/cleaned_data.csv" in required
    assert "data/cleaning_manifest.json" in required
    assert "data/cleaned_data.csv" not in skip_paths


def test_resolve_execute_code_mode_for_data_engineer_without_required_outputs():
    heavy_train = _load_heavy_train_module()
    mode, required, _ = heavy_train._resolve_execute_code_mode({"mode": "data_engineer_cleaning"})
    assert mode == "data_engineer_cleaning"
    assert required == []


def test_resolve_execute_code_mode_for_ml_default():
    heavy_train = _load_heavy_train_module()
    mode, required, skip_paths = heavy_train._resolve_execute_code_mode({"mode": "execute_code"})
    normalized_skip = {str(path).replace("\\", "/") for path in skip_paths}
    assert mode == "execute_code"
    assert required == []
    assert "data/cleaned_data.csv" in normalized_skip


def test_resolve_execute_code_mode_for_ml_uses_contract_required_outputs():
    heavy_train = _load_heavy_train_module()
    mode, required, _ = heavy_train._resolve_execute_code_mode(
        {
            "mode": "execute_code",
            "required_outputs": ["reports/summary.json", "data/metrics.json"],
        }
    )
    assert mode == "execute_code"
    assert required == ["reports/summary.json", "data/metrics.json"]


def test_collect_output_files_respects_skip_paths(tmp_path):
    heavy_train = _load_heavy_train_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw.csv").write_text("a\n1\n", encoding="utf-8")
    (data_dir / "cleaned_data.csv").write_text("a\n1\n", encoding="utf-8")
    (data_dir / "cleaning_manifest.json").write_text("{}", encoding="utf-8")

    collected_ml = heavy_train._collect_output_files(
        str(tmp_path),
        skip_paths={"data/cleaned_data.csv"},
    )
    collected_de = heavy_train._collect_output_files(
        str(tmp_path),
        skip_paths={"data/cleaned_full.csv"},
    )

    assert "data/cleaned_data.csv" not in collected_ml
    assert "data/cleaned_data.csv" in collected_de


def test_execute_code_mode_fails_when_required_outputs_missing(monkeypatch):
    heavy_train = _load_heavy_train_module()

    def _fake_download(_uri, path):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.name == "ml_script.py":
            path_obj.write_text("print('ok')\n", encoding="utf-8")
        else:
            path_obj.write_text("a\n1\n", encoding="utf-8")

    captured_json = []

    monkeypatch.setattr(heavy_train, "_download_to_path", _fake_download)
    monkeypatch.setattr(heavy_train, "_download_support_files", lambda _support, _work_dir: None)
    monkeypatch.setattr(heavy_train, "_run_script", lambda _script, _work_dir: (0, "ok", ""))
    monkeypatch.setattr(heavy_train, "_collect_output_files", lambda _work_dir, skip_paths=None: [])
    monkeypatch.setattr(heavy_train, "_write_file_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        heavy_train,
        "_write_json_output",
        lambda payload, _output_uri, rel_path: captured_json.append((rel_path, payload)),
    )

    payload = {
        "mode": "execute_code",
        "dataset_uri": "gs://bucket/input.csv",
        "data_path": "data/raw.csv",
        "code_uri": "gs://bucket/script.py",
        "required_outputs": ["data/metrics.json"],
    }

    with pytest.raises(RuntimeError, match="missing required outputs"):
        heavy_train.execute_code_mode(payload, "gs://bucket/output/", "run-test")

    assert any(rel == "error.json" for rel, _ in captured_json)
    error_payload = next((item for rel, item in captured_json if rel == "error.json"), {})
    assert error_payload.get("error") == "missing_required_outputs"


def test_execute_code_mode_uploads_required_outputs_outside_default_roots(monkeypatch):
    heavy_train = _load_heavy_train_module()

    def _fake_download(_uri, path):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.name == "ml_script.py":
            path_obj.write_text("print('ok')\n", encoding="utf-8")
        else:
            path_obj.write_text("a\n1\n", encoding="utf-8")

    def _fake_run(_script, work_dir):
        models_dir = Path(work_dir) / "models"
        reports_dir = Path(work_dir) / "reports"
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "best_model.joblib").write_text("binary", encoding="utf-8")
        (reports_dir / "model_card.json").write_text("{}", encoding="utf-8")
        return 0, "ok", ""

    uploaded_files = []
    status_payload = {}

    monkeypatch.setattr(heavy_train, "_download_to_path", _fake_download)
    monkeypatch.setattr(heavy_train, "_download_support_files", lambda _support, _work_dir: None)
    monkeypatch.setattr(heavy_train, "_run_script", _fake_run)
    monkeypatch.setattr(heavy_train, "_collect_output_files", lambda _work_dir, skip_paths=None: [])
    monkeypatch.setattr(
        heavy_train,
        "_write_file_output",
        lambda _local, _output, rel: uploaded_files.append(rel),
    )

    def _capture_json(payload, _output_uri, rel_path):
        if rel_path == "status.json":
            status_payload.update(payload)

    monkeypatch.setattr(heavy_train, "_write_json_output", _capture_json)

    payload = {
        "mode": "execute_code",
        "dataset_uri": "gs://bucket/input.csv",
        "data_path": "data/raw.csv",
        "code_uri": "gs://bucket/script.py",
        "required_outputs": [
            "models/best_model.joblib",
            "reports/model_card.json",
        ],
    }

    exit_code = heavy_train.execute_code_mode(payload, "gs://bucket/output/", "run-test")

    assert exit_code == 0
    assert "models/best_model.joblib" in uploaded_files
    assert "reports/model_card.json" in uploaded_files
    assert status_payload.get("required_outputs_missing") == []
    assert "models/best_model.joblib" in (status_payload.get("uploaded_outputs") or [])
    assert "reports/model_card.json" in (status_payload.get("uploaded_outputs") or [])
