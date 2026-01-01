import argparse
import os
import sys
import zipfile


def _find_latest_run(runs_dir: str) -> str | None:
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
    zip_name = f"run_{run_id}.zip"
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

    print(f"Created {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
