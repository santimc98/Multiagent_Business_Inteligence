"""
Tests for cloudrun_launcher required artifacts validation.

Ensures that:
1. When required artifacts are missing, status is "error"
2. Missing artifacts are listed in the response
3. GCS listing is included for diagnostics when artifacts are missing
"""
from unittest.mock import patch


def test_missing_required_artifacts_marks_error():
    """When required artifacts are not downloaded, status should be 'error'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    # Mock all external dependencies
    with patch("src.utils.cloudrun_launcher._ensure_cli") as mock_cli, \
         patch("src.utils.cloudrun_launcher._gsutil_cp") as mock_cp, \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        # Setup mocks
        mock_cli.return_value = None
        mock_cp.return_value = None
        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # No files exist in GCS output
        mock_exists.return_value = False
        mock_ls.return_value = []

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "error"
        assert "missing_artifacts" in result
        assert set(result["missing_artifacts"]) == {"metrics.json", "scored_rows.csv"}
        assert result["error"]["error"] == "missing_required_artifacts"


def test_partial_artifacts_marks_error():
    """When only some required artifacts are downloaded, status should be 'error'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # Only metrics.json exists
        def exists_side_effect(uri, _):
            return "metrics.json" in uri
        mock_exists.side_effect = exists_side_effect
        mock_ls.return_value = ["gs://bucket/outputs/test_run/metrics.json"]

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
                "alignment_check.json": "data/alignment_check.json",
            },
            required_artifacts=["metrics.json", "scored_rows.csv", "alignment_check.json"],
        )

        assert result["status"] == "error"
        assert "scored_rows.csv" in result["missing_artifacts"]
        assert "alignment_check.json" in result["missing_artifacts"]
        assert "metrics.json" not in result["missing_artifacts"]


def test_all_artifacts_present_marks_success():
    """When all required artifacts are downloaded, status should be 'success'."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls"), \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # All required files exist (except error.json)
        def exists_side_effect(uri, _):
            return any(name in uri for name in ["metrics.json", "scored_rows.csv"])
        mock_exists.side_effect = exists_side_effect

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "success"
        assert result["missing_artifacts"] == []


def test_no_required_artifacts_legacy_behavior():
    """When required_artifacts is None, legacy behavior (no artifact check) applies."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls"), \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True

        # No files exist but no required_artifacts specified
        mock_exists.return_value = False

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
            },
            # required_artifacts not specified (None)
        )

        # Without required_artifacts, no artifact check happens
        assert result["status"] == "success"
        assert result["missing_artifacts"] == []


def test_stale_error_json_is_ignored_when_status_ok_and_artifacts_present():
    """Stale error.json must not fail a run if status.json is ok and artifacts are complete."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("src.utils.cloudrun_launcher._gsutil_cat") as mock_cat, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True
        mock_ls.return_value = []

        def exists_side_effect(uri, _):
            return any(name in uri for name in ["metrics.json", "scored_rows.csv", "status.json", "error.json"])

        def cat_side_effect(uri, _):
            if uri.endswith("status.json"):
                return '{"ok": true, "run_id": "test_run"}'
            if uri.endswith("error.json"):
                return '{"ok": false, "error": "timeout_from_previous_attempt"}'
            return "{}"

        mock_exists.side_effect = exists_side_effect
        mock_cat.side_effect = cat_side_effect

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
                "status.json": "artifacts/heavy_status.json",
                "error.json": "artifacts/heavy_error.json",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "success"
        assert result["error"] is None
        assert isinstance(result["error_raw"], dict)
        assert result["status_ok"] is True
        assert result["status_arbitration"]["applied"] is True
        assert result["status_arbitration"]["ignored_error_payload"] is True


def test_attempt_scopes_output_uri():
    """Attempt id should scope output URI to avoid cross-attempt residue."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.return_value = ("stdout", "stderr", "update-env-vars")
        mock_path_exists.return_value = True
        mock_exists.return_value = False
        mock_ls.return_value = []

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={},
            attempt_id=3,
        )

        assert result["output_uri"].endswith("/outputs/test_run/attempt_3/")
        assert result["attempt_id"] == 3


def test_job_failure_is_overridden_when_status_ok_and_artifacts_present():
    """A launcher transport failure should be downgraded if output contract proves success."""
    from src.utils.cloudrun_launcher import launch_heavy_runner_job, CloudRunLaunchError

    with patch("src.utils.cloudrun_launcher._ensure_cli"), \
         patch("src.utils.cloudrun_launcher._gsutil_cp"), \
         patch("src.utils.cloudrun_launcher._gsutil_exists") as mock_exists, \
         patch("src.utils.cloudrun_launcher._gsutil_ls") as mock_ls, \
         patch("src.utils.cloudrun_launcher._run_gcloud_job_execute") as mock_execute, \
         patch("src.utils.cloudrun_launcher._gsutil_cat") as mock_cat, \
         patch("os.path.exists") as mock_path_exists, \
         patch("os.makedirs"):

        mock_execute.side_effect = CloudRunLaunchError("Command failed: timeout")
        mock_path_exists.return_value = True
        mock_ls.return_value = []

        def exists_side_effect(uri, _):
            return any(name in uri for name in ["metrics.json", "scored_rows.csv", "status.json"])

        def cat_side_effect(uri, _):
            if uri.endswith("status.json"):
                return '{"ok": true, "run_id": "test_run"}'
            return "{}"

        mock_exists.side_effect = exists_side_effect
        mock_cat.side_effect = cat_side_effect

        result = launch_heavy_runner_job(
            run_id="test_run",
            request={"dataset_uri": "gs://bucket/test.csv"},
            dataset_path="data/test.csv",
            bucket="test-bucket",
            job="test-job",
            region="us-central1",
            download_map={
                "metrics.json": "data/metrics.json",
                "scored_rows.csv": "data/scored_rows.csv",
                "status.json": "artifacts/heavy_status.json",
            },
            required_artifacts=["metrics.json", "scored_rows.csv"],
        )

        assert result["status"] == "success"
        assert result["job_failed"] is False
        assert result["job_failed_raw"] is True
        assert result["status_arbitration"]["applied"] is True
        assert result["status_arbitration"]["ignored_job_failure"] is True
