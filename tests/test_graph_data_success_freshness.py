import os
import time

from src.graph.graph import check_data_success


def test_check_data_success_rejects_stale_cleaned_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = data_dir / "cleaned_data.csv"
    cleaned_path.write_text("a,b\n1,2\n", encoding="utf-8")

    run_start_epoch = time.time()
    stale_time = run_start_epoch - 10
    os.utime(cleaned_path, (stale_time, stale_time))

    state = {
        "error_message": None,
        "cleaned_data_preview": "ok",
        "run_start_epoch": run_start_epoch,
    }

    assert check_data_success(state) == "failed"
