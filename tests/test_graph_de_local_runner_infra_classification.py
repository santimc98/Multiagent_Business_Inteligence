from unittest.mock import patch

from src.graph import graph as graph_module


def test_de_local_runner_missing_input_uri_is_classified_as_infra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_EXECUTION_MODE", "local")

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    state = {
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
        }
    }

    fake_heavy_result = {
        "status": "error",
        "downloaded": {},
        "missing_artifacts": ["data/cleaned_data.csv", "data/cleaning_manifest.json"],
        "error": {"error": "required_artifacts_missing"},
        "job_failed": True,
        "job_error": (
            "local_runner_exit_code=1 | tail=ERROR: Failed to load input JSON: "
            "[Errno 2] No such file or directory: '...request.json'"
        ),
        "job_error_raw": "ERROR: Failed to load input JSON: ...request.json",
        "job_stdout": "",
        "job_stderr": "ERROR: Failed to load input JSON: ...request.json",
        "status_ok": False,
    }

    with patch("src.graph.graph.launch_local_runner_job", return_value=fake_heavy_result):
        result = graph_module._execute_data_engineer_via_heavy_runner(
            state=state,
            code="print('ok')",
            csv_path=str(csv_path),
            csv_sep=",",
            csv_decimal=".",
            csv_encoding="utf-8",
            heavy_cfg={"bucket": "b", "job": "j", "region": "r"},
            run_id=None,
            attempt_id=1,
            reason="local_runner_mode",
        )

    assert result["ok"] is False
    assert result["error_kind_hint"] == "infra_input_uri_missing"
    assert str(result["error_details"]).startswith("LOCAL_RUNNER_INPUT_URI_MISSING")
