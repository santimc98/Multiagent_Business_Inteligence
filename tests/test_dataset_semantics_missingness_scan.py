import pytest

from src.utils.dataset_semantics import infer_dataset_semantics


def test_target_missingness_scan_chunks(tmp_path):
    rows = ["feature,target"]
    rows.extend([f"{i},{i % 2}" for i in range(100)])
    rows.extend([f"{i}," for i in range(100, 150)])
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    semantics = infer_dataset_semantics(
        csv_path=str(csv_path),
        dialect={"sep": ",", "decimal": ".", "encoding": "utf-8"},
        contract_min={"outcome_columns": ["target"]},
        max_sample_rows=2000,
    )

    target_info = semantics.get("target_analysis", {})
    assert target_info.get("partial_label_detected") is True
    assert target_info.get("target_total_count_exact") == 150
    assert target_info.get("target_missing_count_exact") == 50
    assert target_info.get("target_null_frac_exact") == pytest.approx(50 / 150)
