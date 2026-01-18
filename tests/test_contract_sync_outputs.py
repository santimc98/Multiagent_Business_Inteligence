from src.agents.execution_planner import _sync_execution_contract_outputs


def test_sync_execution_contract_outputs_uses_contract_min() -> None:
    contract = {
        "required_outputs": ["Priority ranking"],
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
    }
    contract_min = {
        "required_outputs": ["data/scored_rows.csv", "data/alignment_check.json"],
        "artifact_requirements": {
            "required_files": [
                {"path": "data/scored_rows.csv"},
                {"path": "data/alignment_check.json"},
            ]
        },
    }

    synced = _sync_execution_contract_outputs(contract, contract_min)

    required_outputs = synced.get("required_outputs", [])
    assert "Priority ranking" not in required_outputs
    assert "data/scored_rows.csv" in required_outputs
    assert "data/alignment_check.json" in required_outputs

    required_files = synced.get("artifact_requirements", {}).get("required_files", [])
    paths = [item.get("path") if isinstance(item, dict) else item for item in required_files]
    assert "data/metrics.json" in paths
    assert "data/scored_rows.csv" in paths
    assert "data/alignment_check.json" in paths
