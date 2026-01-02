import os
from pathlib import Path

from src.utils.run_storage import (
    init_run_dir,
    finalize_run,
    apply_retention,
)


def test_latest_is_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    latest = tmp_path / "runs" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    dummy = latest / "dummy.txt"
    dummy.write_text("stale", encoding="utf-8")
    run_dir = init_run_dir("abc123", started_at="2025-01-01T00:00:00")
    assert not dummy.exists()
    assert (latest / "run_id.txt").exists()
    assert (Path(run_dir) / "run_manifest.json").exists()


def test_archive_on_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = init_run_dir("fail123", started_at="2025-01-01T00:00:00")
    (Path(run_dir) / "dummy.txt").write_text("x", encoding="utf-8")
    finalize_run("fail123", status_final="FAIL", state={})
    archive = tmp_path / "runs" / "archive" / "run_fail123.zip"
    assert archive.exists()


def test_no_archive_on_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_run_dir("pass123", started_at="2025-01-01T00:00:00")
    finalize_run("pass123", status_final="PASS", state={})
    archive_dir = tmp_path / "runs" / "archive"
    if archive_dir.exists():
        assert not any(archive_dir.glob("run_pass123.zip"))


def test_retention_keeps_last_n(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive_dir = tmp_path / "runs" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for idx in range(7):
        path = archive_dir / f"run_{idx}.zip"
        path.write_bytes(b"zip")
        zips.append(path)
    for idx, path in enumerate(zips):
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + idx))
    apply_retention(keep_last=5, archive_dir=str(archive_dir))
    remaining = list(archive_dir.glob("*.zip"))
    assert len(remaining) == 5
