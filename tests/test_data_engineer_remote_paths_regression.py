from pathlib import Path


def test_remote_paths_are_bound_before_download_in_run_data_engineer() -> None:
    src = Path("src/graph/graph.py").read_text(encoding="utf-8")
    fn_start = src.index("def run_data_engineer(state: AgentState) -> AgentState:")
    fn_end = src.index("\ndef check_data_success(state: AgentState)", fn_start)
    block = src[fn_start:fn_end]

    assign_clean = 'remote_cleaned_rel = local_cleaned_path.lstrip("/").replace("\\\\", "/")'
    assign_manifest = 'remote_manifest_rel = local_manifest_path.lstrip("/").replace("\\\\", "/")'
    if_exec_error = "if execution.error:"
    use_clean = 'csv_content = safe_download_bytes(sandbox, f"{run_root}/{remote_cleaned_rel}")'
    use_manifest = 'manifest_content = safe_download_bytes(sandbox, f"{run_root}/{remote_manifest_rel}")'

    assert assign_clean in block
    assert assign_manifest in block
    assert if_exec_error in block
    assert use_clean in block
    assert use_manifest in block

    assert block.index(assign_clean) < block.index(if_exec_error)
    assert block.index(assign_manifest) < block.index(if_exec_error)
    assert block.index(assign_clean) < block.index(use_clean)
    assert block.index(assign_manifest) < block.index(use_manifest)
