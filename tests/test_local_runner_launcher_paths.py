import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.local_runner_launcher import launch_local_runner_job


def test_local_runner_writes_absolute_request_uris(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("a,b\n1,2\n", encoding="utf-8")
    support_path = tmp_path / "support.json"
    support_path.write_text('{"ok": true}', encoding="utf-8")

    with patch(
        "src.utils.local_runner_launcher.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="simulated failure"),
    ):
        result = launch_local_runner_job(
            run_id="r-local-abs",
            request={"mode": "data_engineer_cleaning"},
            dataset_path=str(dataset_path),
            bucket="unused",
            job="unused",
            region="unused",
            code_text="print('ok')",
            data_path="data/raw.csv",
            support_files=[{"local_path": str(support_path), "path": "data/support.json"}],
            attempt_id=2,
            stage_namespace="data_engineer",
        )

    assert os.path.isabs(result["input_uri"])
    with open(result["input_uri"], "r", encoding="utf-8") as f_req:
        payload = json.load(f_req)
    assert os.path.isabs(payload["code_uri"])
    assert payload.get("support_files")
    assert os.path.isabs(payload["support_files"][0]["uri"])
    assert os.path.isabs(payload["output_uri"].rstrip("\\/"))
