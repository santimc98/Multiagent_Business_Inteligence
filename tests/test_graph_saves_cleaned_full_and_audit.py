import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest
from unittest.mock import patch

from src.graph.graph import run_data_engineer


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


def _mock_command_ok(stdout=""):
    return SimpleNamespace(exit_code=0, stdout=stdout)


def test_graph_saves_cleaned_full_and_audit(tmp_workdir, monkeypatch):
    raw_path = tmp_workdir / "raw.csv"
    raw_path.write_text("col1,col2,target\n1,2,3\n4,5,9\n", encoding="utf-8")

    cleaned_bytes = b"a,b,target\n1,2,3\n4,5,9\n"
    manifest_bytes = json.dumps(
        {"output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
    ).encode("utf-8")

    # V4.1: Mock cleaning reviewer to pass (contract-strict mode rejects without cleaning_gates)
    mock_cleaning_result = {
        "status": "APPROVED",
        "feedback": "Test mock approval",
        "failed_checks": [],
        "required_fixes": [],
        "warnings": [],
        "hard_failures": [],
        "soft_failures": [],
        "contract_source_used": "cleaning_view",
    }
    with patch("src.graph.graph.Sandbox") as MockSandbox, \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])), \
         patch("src.graph.graph.data_engineer.generate_cleaning_script", return_value="print('clean')"), \
         patch("src.graph.graph.cleaning_reviewer.review_cleaning", return_value=mock_cleaning_result), \
         patch.dict(os.environ, {"E2B_API_KEY": "dummy", "DEEPSEEK_API_KEY": "dummy", "GOOGLE_API_KEY": "dummy"}):

        mock_instance = MockSandbox.create.return_value.__enter__.return_value
        mock_instance.commands.run.return_value = _mock_command_ok()
        mock_instance.run_code.return_value = SimpleNamespace(
            logs=SimpleNamespace(stdout=["ok"], stderr=[]),
            error=None
        )
        mock_instance.files.read.side_effect = [cleaned_bytes, manifest_bytes]

        state = {
            "selected_strategy": {"title": "t", "analysis_type": "regression", "required_columns": ["a", "b", "target"]},
            "business_objective": "",
            "csv_path": str(raw_path),
            "csv_encoding": "utf-8",
            "csv_sep": ",",
            "csv_decimal": ".",
            "data_summary": "",
            "leakage_audit_summary": "",
            "execution_contract": {"contract_version": "4.1"},
            "execution_contract_diagnostics": {
                "validation": {"accepted": True, "status": "ok"},
                "summary": {"accepted": True},
            },
        }

        result = run_data_engineer(state)

        assert os.path.exists("data/cleaned_full.csv")
        assert os.path.exists("data/leakage_audit.json")
        assert "leakage_audit_summary" in result
