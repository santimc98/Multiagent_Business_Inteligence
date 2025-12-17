import os
from types import SimpleNamespace
from unittest.mock import patch, mock_open

import pytest

from src.graph.graph import execute_code


def test_ml_code_artifact_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    (tmp_path / "data" / "cleaned_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    state = {
        "generated_code": "print('hello world')",
        "execution_output": "",
        "execution_attempt": 1
    }

    with patch("src.graph.graph.Sandbox") as MockSandbox, \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])), \
         patch.dict(os.environ, {"E2B_API_KEY": "dummy"}):

        mock_instance = MockSandbox.create.return_value.__enter__.return_value
        mock_instance.commands.run.return_value = SimpleNamespace(exit_code=0, stdout="")
        mock_instance.run_code.return_value = SimpleNamespace(
            logs=SimpleNamespace(stdout=["ok"], stderr=[]),
            error=None
        )
        mock_instance.files.write.return_value = None

        execute_code(state)

        artifact_path = tmp_path / "artifacts" / "ml_engineer_last.py"
        assert artifact_path.exists()
        content = artifact_path.read_text(encoding="utf-8")
        assert "hello world" in content
