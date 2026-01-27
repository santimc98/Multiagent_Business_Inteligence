import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple


class CloudRunLaunchError(RuntimeError):
    pass


def _ensure_cli(binary: str) -> None:
    if not shutil.which(binary):
        raise CloudRunLaunchError(f"Required CLI not found: {binary}")


def _run_cmd(args: list[str], timeout_s: Optional[int] = None) -> Tuple[str, str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise CloudRunLaunchError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _gsutil_exists(uri: str) -> bool:
    _ensure_cli("gsutil")
    proc = subprocess.run(
        ["gsutil", "-q", "stat", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _gsutil_cp(src: str, dest: str) -> None:
    _ensure_cli("gsutil")
    _run_cmd(["gsutil", "-q", "cp", src, dest])


def _gsutil_cat(uri: str) -> str:
    _ensure_cli("gsutil")
    stdout, _ = _run_cmd(["gsutil", "cat", uri])
    return stdout


def _upload_json_to_gcs(payload: Dict[str, Any], uri: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp_path = tmp.name
    try:
        _gsutil_cp(tmp_path, uri)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().strip("/")
    return prefix


def launch_heavy_runner_job(
    *,
    run_id: str,
    request: Dict[str, Any],
    dataset_path: str,
    bucket: str,
    job: str,
    region: str,
    input_prefix: str = "inputs",
    output_prefix: str = "outputs",
    dataset_prefix: str = "datasets",
    project: Optional[str] = None,
    download_map: Optional[Dict[str, str]] = None,
    wait: bool = True,
) -> Dict[str, Any]:
    _ensure_cli("gcloud")
    _ensure_cli("gsutil")

    input_prefix = _normalize_prefix(input_prefix)
    output_prefix = _normalize_prefix(output_prefix)
    dataset_prefix = _normalize_prefix(dataset_prefix)

    dataset_uri = request.get("dataset_uri")
    if not dataset_uri or not str(dataset_uri).startswith("gs://"):
        if not dataset_path or not os.path.exists(dataset_path):
            raise CloudRunLaunchError("dataset_path missing or does not exist for heavy runner upload")
        dataset_name = os.path.basename(dataset_path)
        dataset_uri = f"gs://{bucket}/{dataset_prefix}/{run_id}/{dataset_name}"
        _gsutil_cp(dataset_path, dataset_uri)
    request["dataset_uri"] = dataset_uri

    input_uri = f"gs://{bucket}/{input_prefix}/{run_id}.json"
    output_uri = f"gs://{bucket}/{output_prefix}/{run_id}/"
    request["output_uri"] = output_uri
    _upload_json_to_gcs(request, input_uri)

    env_vars = f"INPUT_URI={input_uri},OUTPUT_URI={output_uri}"
    cmd = ["gcloud", "run", "jobs", "execute", job, "--region", region, "--set-env-vars", env_vars]
    if project:
        cmd.extend(["--project", project])
    if wait:
        cmd.append("--wait")
    stdout, stderr = _run_cmd(cmd)

    downloaded: Dict[str, str] = {}
    if download_map:
        for filename, local_path in download_map.items():
            if not filename:
                continue
            remote_path = output_uri + filename
            if not _gsutil_exists(remote_path):
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _gsutil_cp(remote_path, local_path)
            downloaded[filename] = local_path

    error_payload: Optional[Dict[str, Any]] = None
    error_uri = output_uri + "error.json"
    if _gsutil_exists(error_uri):
        try:
            error_payload = json.loads(_gsutil_cat(error_uri))
        except Exception:
            error_payload = {"error": "Failed to parse error.json", "raw": _gsutil_cat(error_uri)}

    return {
        "status": "error" if error_payload else "success",
        "input_uri": input_uri,
        "output_uri": output_uri,
        "dataset_uri": dataset_uri,
        "downloaded": downloaded,
        "job_stdout": stdout,
        "job_stderr": stderr,
        "error": error_payload,
    }
