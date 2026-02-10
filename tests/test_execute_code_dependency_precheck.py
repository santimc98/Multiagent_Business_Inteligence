from unittest.mock import patch

from src.graph import graph as graph_module


def test_execute_code_blocks_globally_banned_dependency_before_heavy_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "generated_code": "import pulp\nprint('ok')\n",
        "feedback_history": [],
        "run_id": "dep-precheck-heavy",
    }

    with patch.object(
        graph_module,
        "_get_heavy_runner_config",
        return_value={"bucket": "b", "job": "j", "region": "r"},
    ), patch.object(
        graph_module,
        "_should_use_heavy_runner",
        return_value=(True, "large_dataset"),
    ), patch.object(
        graph_module,
        "launch_heavy_runner_job",
    ) as mock_launch:
        result = graph_module.execute_code(state)

    assert "DEPENDENCY_BLOCKED" in str(result.get("error_message", ""))
    assert "pulp" in str(result.get("error_message", "")).lower()
    assert mock_launch.call_count == 0


def test_execute_code_allows_torch_when_cloudrun_selected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "generated_code": "import torch\nprint('ok')\n",
        "feedback_history": [],
        "run_id": "dep-precheck-cloudrun-allow",
    }

    with patch.object(
        graph_module,
        "_get_heavy_runner_config",
        return_value={"bucket": "b", "job": "j", "region": "r"},
    ), patch.object(
        graph_module,
        "_should_use_heavy_runner",
        return_value=(True, "large_dataset"),
    ), patch.object(
        graph_module,
        "launch_heavy_runner_job",
    ) as mock_launch:
        result = graph_module.execute_code(state)

    # Not blocked by dependency precheck; downstream may fail due missing data context in unit test setup.
    assert "DEPENDENCY_BLOCKED" not in str(result.get("error_message", ""))
    assert mock_launch.call_count == 0
