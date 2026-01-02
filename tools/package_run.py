import argparse
import json
import os
import sys
import zipfile

from src.utils.run_storage import apply_retention


def _latest_dir(runs_dir: str) -> str:
    return os.path.join(runs_dir, "latest")


def _read_status(manifest_path: str) -> str | None:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("status_final")
    except Exception:
        return None


def _read_run_id(manifest_path: str) -> str | None:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("run_id")
    except Exception:
        return None


def _read_latest_run_id(runs_dir: str) -> str | None:
    try:
        with open(os.path.join(_latest_dir(runs_dir), "run_id.txt"), "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def _find_latest_run(runs_dir: str) -> str | None:
    latest_id = _read_latest_run_id(runs_dir)
    if latest_id:
        return latest_id
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        if not os.path.isdir(run_path):
            continue
        manifest = os.path.join(run_path, "run_manifest.json")
        if os.path.exists(manifest):
            ts = os.path.getmtime(manifest)
        else:
            ts = os.path.getmtime(run_path)
        candidates.append((ts, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _zip_run(run_id: str, runs_dir: str) -> str:
    run_dir = os.path.join(runs_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    actual_id = _read_run_id(manifest_path) or run_id
    zip_name = f"run_{actual_id}.zip"
    zip_path = os.path.abspath(zip_name)
    base_dir = os.path.abspath(os.path.dirname(run_dir))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, base_dir)
                zf.write(path, arcname)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a run bundle into a zip file.")
    parser.add_argument("--latest", action="store_true", help="Package the most recent run.")
    parser.add_argument("--run-id", default="", help="Package a specific run id.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory path.")
    parser.add_argument("--archive-on-fail", action="store_true", help="Move zip into runs/archive when status_final != PASS.")
    args = parser.parse_args()

    run_id = args.run_id
    if args.latest:
        run_id = _find_latest_run(args.runs_dir)
    if not run_id:
        print("No run_id specified or found.")
        return 1

    try:
        zip_path = _zip_run(run_id, args.runs_dir)
    except Exception as exc:
        print(f"Failed to package run: {exc}")
        return 1

    if args.archive_on_fail:
        run_dir = os.path.join(args.runs_dir, run_id)
        status = _read_status(os.path.join(run_dir, "run_manifest.json"))
        if status and str(status).upper() != "PASS":
            archive_dir = os.path.join(args.runs_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            dest = os.path.join(archive_dir, os.path.basename(zip_path))
            try:
                os.replace(zip_path, dest)
                zip_path = dest
            except Exception:
                pass
            apply_retention(keep_last=5, archive_dir=archive_dir)

    print(f"Created {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
