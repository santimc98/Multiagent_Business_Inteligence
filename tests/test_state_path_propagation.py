import os

from src.utils.path_resolution import _add_workspace_metadata, _resolve_csv_path_with_base


def test_state_path_propagation_helpers(tmp_path):
    base_cwd = str(tmp_path)
    csv_rel = os.path.join("data", "x.csv")
    expected = os.path.normpath(os.path.abspath(os.path.join(base_cwd, csv_rel)))

    resolved = _resolve_csv_path_with_base(base_cwd, csv_rel)
    assert resolved == expected

    payload = {"data_summary": "ok"}
    state = {
        "csv_path": resolved,
        "_orig_cwd": base_cwd,
        "work_dir": os.path.join(base_cwd, "runs", "abc", "work"),
        "workspace_active": True,
    }
    run_dir = os.path.join(base_cwd, "runs", "abc")
    updated = _add_workspace_metadata(payload, state, base_cwd, run_dir)

    assert updated["csv_path"] == resolved
    assert updated["_orig_cwd"] == base_cwd
    assert updated["work_dir"] == state["work_dir"]
    assert updated["workspace_active"] is True
    assert updated["run_dir"] == run_dir
