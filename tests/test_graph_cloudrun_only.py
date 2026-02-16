from unittest.mock import patch

from src.graph.graph import execute_code


def test_execute_code_requires_cloudrun_config_when_cloudrun_only():
    state = {
        "generated_code": "print('ok')\n",
        "feedback_history": [],
    }
    with patch("src.graph.graph._get_heavy_runner_config", return_value=None), \
         patch("src.graph.graph._get_execution_runtime_mode", return_value="cloudrun"):
        result = execute_code(state)

    assert "CLOUDRUN_REQUIRED" in str(result.get("error_message", ""))
    assert "Cloud Run" in str(result.get("execution_output", ""))
