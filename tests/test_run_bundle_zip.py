import zipfile
from pathlib import Path

from tools.package_run import _zip_run


def test_run_bundle_zip_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "runabc"
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "artifacts" / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "artifacts" / "data" / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")

    zip_path = _zip_run(run_id, runs_dir=str(tmp_path / "runs"))
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    assert all(name.startswith(f"{run_id}/") for name in names)
    assert not any("secret.txt" in name for name in names)
