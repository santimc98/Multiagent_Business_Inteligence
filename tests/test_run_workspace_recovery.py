import os

from src.utils.run_workspace import exit_run_workspace, recover_orphaned_workspace_cwd


def test_recover_orphaned_workspace_cwd_restores_project_root(tmp_path, monkeypatch):
    work_dir = tmp_path / "runs" / "abc123" / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)

    restored = recover_orphaned_workspace_cwd(project_root=str(tmp_path))

    assert restored == os.path.normpath(str(tmp_path))
    assert os.path.normcase(os.getcwd()) == os.path.normcase(os.path.normpath(str(tmp_path)))


def test_exit_run_workspace_uses_fallback_when_orig_cwd_missing(tmp_path, monkeypatch):
    work_dir = tmp_path / "runs" / "runxyz" / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)

    state = {"orig_cwd": str(tmp_path / "missing_dir"), "workspace_active": True}
    exit_run_workspace(state)

    assert os.path.normcase(os.getcwd()) == os.path.normcase(os.path.normpath(str(tmp_path)))
    assert state["workspace_active"] is False
