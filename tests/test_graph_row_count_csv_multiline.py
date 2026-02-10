import csv

from src.graph.graph import _estimate_row_count


def test_estimate_row_count_handles_multiline_csv_cells(tmp_path):
    csv_path = tmp_path / "multiline.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "score"])
        writer.writerow([1, "hello\nworld", 0.9])
        writer.writerow([2, "single", 0.2])

    rows = _estimate_row_count(str(csv_path), encoding="utf-8", sep=",")
    assert rows == 2
